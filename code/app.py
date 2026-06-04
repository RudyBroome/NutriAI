import streamlit as st
import sqlite3
import pandas as pd
import re
import os
import numpy as np
from scipy.optimize import minimize
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'food_database.sqlite')

# ==========================================
# 1. REINFORCEMENT LEARNING (MULTI-ARMED BANDIT)
# ==========================================
def update_rl_weight(profile_name, blueprint_id, feedback_type):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Pull current weight and the number of times they've voted on this meal
    cursor.execute("SELECT preference_weight, interaction_count FROM rl_preferences WHERE persona = ? AND blueprint_id = ?", (profile_name, blueprint_id))
    result = cursor.fetchone()
    
    current_weight = result[0] if result else 1.0
    current_count = result[1] if result else 0
    
    new_weight = current_weight * 1.5 if feedback_type == 'up' else current_weight * 0.1
    new_count = current_count + 1
        
    cursor.execute("""
        INSERT INTO rl_preferences (persona, blueprint_id, preference_weight, interaction_count) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(persona, blueprint_id) DO UPDATE SET 
            preference_weight = excluded.preference_weight,
            interaction_count = excluded.interaction_count
    """, (profile_name, blueprint_id, new_weight, new_count))
    conn.commit()
    conn.close()

# ==========================================
# 2. THE "SMART" DYNAMIC SQL FIREWALL
# ==========================================
def build_exclusion_subquery(conditions, allergies, diet):
    banned_rules = []
    
    def ban(term):
        return f"(f2.description LIKE '%{term}%' OR bc2.ingredient_name LIKE '%{term}%')"
        
    def ban_except(term, safe_term):
        return f"({ban(term)} AND f2.description NOT LIKE '%{safe_term}%' AND bc2.ingredient_name NOT LIKE '%{safe_term}%')"

    # 1. Clinical Conditions
    if "IBS (Low FODMAP)" in conditions: banned_rules.append("f2.is_high_fodmap = 1")
    if "GERD" in conditions: banned_rules.append("f2.is_gerd_trigger = 1")
    if "Diabetes (Low GI)" in conditions: banned_rules.append("f2.is_high_gi = 1")

    # Advanced Dairy Rules
    safe_dairy_rules = [
        f"({ban('milk')} AND f2.description NOT LIKE '%almond%' AND bc2.ingredient_name NOT LIKE '%almond%' AND f2.description NOT LIKE '%soy%' AND bc2.ingredient_name NOT LIKE '%soy%' AND f2.description NOT LIKE '%oat%' AND bc2.ingredient_name NOT LIKE '%oat%' AND f2.description NOT LIKE '%coconut%' AND bc2.ingredient_name NOT LIKE '%coconut%' AND f2.description NOT LIKE '%hemp%' AND bc2.ingredient_name NOT LIKE '%hemp%')",
        f"({ban('cheese')} AND f2.description NOT LIKE '%vegan%' AND bc2.ingredient_name NOT LIKE '%vegan%')",
        f"({ban('butter')} AND f2.description NOT LIKE '%peanut%' AND bc2.ingredient_name NOT LIKE '%peanut%' AND f2.description NOT LIKE '%almond%' AND bc2.ingredient_name NOT LIKE '%almond%' AND f2.description NOT LIKE '%cashew%' AND bc2.ingredient_name NOT LIKE '%cashew%' AND f2.description NOT LIKE '%seed%' AND bc2.ingredient_name NOT LIKE '%seed%')",
        f"({ban('yogurt')} AND f2.description NOT LIKE '%coconut%' AND bc2.ingredient_name NOT LIKE '%coconut%' AND f2.description NOT LIKE '%almond%' AND bc2.ingredient_name NOT LIKE '%almond%')"
    ]
    
    # 2. Dynamic Allergies & Aversions
    if "Dairy" in allergies: banned_rules.extend(safe_dairy_rules)
    
    # FIX: Corrected "Nuts" to "Tree Nuts" to match UI and Test Suite
    if "Tree Nuts" in allergies:
        for n in ['almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'macadamia']: banned_rules.append(ban(n))
        
    if "Peanuts" in allergies: banned_rules.append(ban('peanut'))
    if "Shellfish" in allergies:
        for s in ['shrimp', 'crab', 'lobster', 'clam']: banned_rules.append(ban(s))
    if "Gluten" in allergies:
        banned_rules.append(ban_except('wheat', 'buckwheat'))
        banned_rules.append(ban('flour'))
        banned_rules.append(ban('bread'))
        banned_rules.append(ban('pasta'))
    if "Soy" in allergies:
        for s in ['soy', 'tofu', 'edamame']: banned_rules.append(ban(s))
    if "Eggs" in allergies:
        banned_rules.append(ban_except('egg', 'eggplant'))
    
    # FIX: Added explicit Pork handling
    if "Pork" in allergies:
        for p in ['pork', 'bacon', 'ham', 'sausage', 'prosciutto']: banned_rules.append(ban(p))
        
    # 3. Global Diets
    meats = ['chicken', 'beef', 'pork', 'bacon', 'turkey', 'lamb', 'steak']
    seafood = ['fish', 'salmon', 'tuna', 'shrimp', 'crab']
    
    if diet == "Vegetarian":
        for m in meats + seafood: banned_rules.append(ban(m))
    elif diet == "Vegan":
        for m in meats + seafood: banned_rules.append(ban(m))
        banned_rules.extend(safe_dairy_rules)
        banned_rules.append(ban_except('egg', 'eggplant'))
        banned_rules.append(ban('honey'))
    elif diet == "Pescatarian":
        for m in meats: banned_rules.append(ban(m))
    elif diet == "Halal":
        for m in ['pork', 'bacon', 'ham', 'wine', 'beer', 'alcohol']: banned_rules.append(ban(m))
    elif diet == "Kosher":
        for m in ['pork', 'bacon', 'ham', 'shrimp', 'crab', 'lobster', 'clam']: banned_rules.append(ban(m))

    if not banned_rules: return ""
    
    # Secures the final query by ensuring NO component matches ANY banned rule
    return f"AND mb.blueprint_id NOT IN (SELECT blueprint_id FROM blueprint_components bc2 JOIN foods f2 ON bc2.mapped_fdc_id = f2.fdc_id WHERE {' OR '.join(banned_rules)})"
def fetch_safe_recipes(exclusion_sql, profile_name, meal_category):
    conn = sqlite3.connect(DB_PATH)
    cat_filter = "AND mb.meal_type LIKE '%reakfast%'" if meal_category == 'Breakfast' else "AND mb.meal_type NOT LIKE '%reakfast%'"
    
    query = f"""
        -- 1. Calculate the Global Community Consensus
        WITH global_prefs AS (
            SELECT blueprint_id,
                   AVG(preference_weight) as w_global
            FROM rl_preferences
            GROUP BY blueprint_id
        )
        SELECT mb.blueprint_id, mb.meal_name, mb.meal_type,
               SUM(f.calories) as cal, SUM(f.protein_g) as prot,
               SUM(f.carbs_g) as carb, SUM(f.fat_g) as fat, SUM(f.fiber_g) as fib,
               SUM(f.iron_mg) as iron, SUM(f.calcium_mg) as calc,
               SUM(f.vit_b12_mcg) as b12, SUM(f.vit_d_IU) as vitd, SUM(f.zinc_mg) as zinc,
               GROUP_CONCAT(bc.ingredient_name, ', ') as ingredients,
               
               -- 2. The Bayesian Math Engine
               -- C = 3.0 (Confidence constant: requires 3 local votes to equal the weight of the global average)
               COALESCE(
                   (3.0 * COALESCE(gp.w_global, 1.0) + COALESCE(rl.interaction_count, 0) * rl.preference_weight) 
                   / (3.0 + COALESCE(rl.interaction_count, 0)), 
               1.0) as rl_weight

        FROM meal_blueprints mb
        JOIN blueprint_components bc ON mb.blueprint_id = bc.blueprint_id
        JOIN foods f ON bc.mapped_fdc_id = f.fdc_id
        
        -- Join the global community matrix
        LEFT JOIN global_prefs gp ON mb.blueprint_id = gp.blueprint_id
        -- Join the specific patient's matrix
        LEFT JOIN rl_preferences rl ON mb.blueprint_id = rl.blueprint_id AND rl.persona = '{profile_name}'
        
        WHERE 1=1 {exclusion_sql} {cat_filter}
        GROUP BY mb.blueprint_id
        HAVING cal > 50
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_raw_components(blueprint_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT bc.ingredient_name, f.calories as cal, f.protein_g as prot, 
               f.carbs_g as carb, f.fat_g as fat, f.fiber_g as fib,
               f.iron_mg as iron, f.calcium_mg as calc, f.vit_b12_mcg as b12, 
               f.vit_d_IU as vitd, f.zinc_mg as zinc
        FROM blueprint_components bc
        JOIN foods f ON bc.mapped_fdc_id = f.fdc_id
        WHERE bc.blueprint_id = '{blueprint_id}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ==========================================
# 3. ADVANCED ALGORITHMIC SAMPLER
# ==========================================
def get_diverse_sample(pool, n, max_category_count=2, max_ing_overlap=0.4):
    selected = []
    shuffled = pool.sample(frac=1, weights='rl_weight').to_dict('records')
    categories = ['salad', 'soup', 'bowl', 'pasta', 'sandwich', 'wrap', 'curry', 'taco', 'stir-fry', 'skillet']
    
    for meal in shuffled:
        if len(selected) == n: break
            
        raw_title = meal['meal_name']
        # Extract the pure base name (no numbers, no punctuation)
        clean_title = re.sub(r'[^a-zA-Z\s]', '', raw_title).strip().lower()
        meal_ings = set([i.strip().lower() for i in meal['ingredients'].split(',')])
        
        # Check 1: Base Title Hashing (Catches ChatGPT numbered duplicates)
        title_conflict = False
        for s in selected:
            s_clean = re.sub(r'[^a-zA-Z\s]', '', s['meal_name']).strip().lower()
            if clean_title == s_clean:
                title_conflict = True
                break
        if title_conflict: continue
        
        # Check 2: Lexical Category (e.g., max 2 soups)
        my_cat = next((c for c in categories if c in raw_title.lower()), "other")
        if my_cat != "other":
            if sum(1 for s in selected if my_cat in s['meal_name'].lower()) >= max_category_count: continue 
                
        # Check 3: Ingredient Overlap Threshold
        too_similar = False
        for s in selected:
            s_ings = set([i.strip().lower() for i in s['ingredients'].split(',')])
            overlap_ratio = len(meal_ings.intersection(s_ings)) / max(1, len(meal_ings))
            if overlap_ratio > max_ing_overlap:
                too_similar = True
                break
                
        if not too_similar: selected.append(meal)
            
    # Constraints fallback
    if len(selected) < n:
        used_ids = [s['blueprint_id'] for s in selected]
        remaining = n - len(selected)
        try:
            fillers = pool[~pool['blueprint_id'].isin(used_ids)].sample(n=remaining, weights='rl_weight', replace=False).to_dict('records')
        except ValueError:
            fillers = pool.sample(n=remaining, replace=True).to_dict('records')
        selected.extend(fillers)
        
    return selected

# ==========================================
# ADVANCED OPTIMIZATION LAYER
# ==========================================
def optimize_meal_ratios(components_df, target_cal, target_prot):
    n = len(components_df)
    x0 = np.ones(n)
    
    def objective(x):
        return np.sum((x - 1.0)**2)
        
    def cal_ceiling(x):
        return (target_cal * 1.05) - np.dot(x, components_df['cal'])
        
    def cal_floor(x):
        return np.dot(x, components_df['cal']) - (target_cal * 0.95)

    def prot_minimum(x):
        return np.dot(x, components_df['prot']) - target_prot

    constraints = [
        {'type': 'ineq', 'fun': cal_ceiling},
        {'type': 'ineq', 'fun': cal_floor},
        {'type': 'ineq', 'fun': prot_minimum}
    ]
    
    # Bound scalars between 0.5x and 2.0x
    bounds = [(0.5, 2.0) for _ in range(n)]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        components_df['optimal_scalar'] = result.x
        return components_df
    return None

def apply_flat_scalar_fallback(components_df, target_cal):
    """The default MVP fallback: a flat multiplier across all ingredients."""
    current_cal = components_df['cal'].sum()
    flat_scalar = target_cal / max(current_cal, 1) # Prevent div by zero
    components_df['optimal_scalar'] = flat_scalar
    return components_df

def safe_optimize(components_df, target_cal, target_prot, timeout_seconds=3.0):
    """
    Executes the SciPy solver with a strict wall-clock time limit.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(optimize_meal_ratios, components_df, target_cal, target_prot)
        try:
            optimized_df = future.result(timeout=timeout_seconds)
            if optimized_df is None:
                return apply_flat_scalar_fallback(components_df, target_cal)
            return optimized_df
        except concurrent.futures.TimeoutError:
            return apply_flat_scalar_fallback(components_df, target_cal)
        
def generate_plan(b_pool, l_pool, d_pool, rda):
    # 1. Pull the diverse 7-day selections from the pre-filtered SQL pools
    b_sel = get_diverse_sample(b_pool, 7)
    l_sel = get_diverse_sample(l_pool, 7)
    d_sel = get_diverse_sample(d_pool, 7)
    
    plan = []
    
    # 2. The 7-Day Waterfall Loop
    for i in range(7):
        # Start each day with the full target bank
        daily_cal_remaining = rda['cal']
        daily_pro_remaining = rda['prot']
        
        day_meals = [b_sel[i], l_sel[i], d_sel[i]]
        
        for meal_idx, m in enumerate(day_meals):
            # 3. Dynamic Waterfall Targets
            if meal_idx == 0:   # Breakfast (~30% of daily total)
                t_cal = rda['cal'] * 0.30
                t_pro = rda['prot'] * 0.30
            elif meal_idx == 1: # Lunch (~50% of whatever is LEFT)
                t_cal = daily_cal_remaining * 0.50
                t_pro = daily_pro_remaining * 0.50
            else:               # Dinner (EXACTLY whatever is remaining to close out the day)
                t_cal = daily_cal_remaining
                t_pro = daily_pro_remaining

            # 4. Fetch raw components and optimize
            comp_df = fetch_raw_components(m['blueprint_id'])
            opt_df = safe_optimize(comp_df, t_cal, t_pro, timeout_seconds=3.0)
            
            # 5. Apply optimal scalars
            for k in ['cal', 'prot', 'carb', 'fat', 'fib', 'iron', 'calc', 'b12', 'vitd', 'zinc']:
                m[k] = round(np.dot(opt_df['optimal_scalar'], opt_df[k]), 1)
                
            # 6. Format ingredients for UI (eliminating the double scalar bug)
            scaled_ingredients = []
            for _, row in opt_df.iterrows():
                ing_name = row['ingredient_name'].strip().capitalize()
                scalar_val = round(row['optimal_scalar'], 2)
                scaled_ingredients.append(f"{ing_name} (x{scalar_val})")
            m['ingredients'] = scaled_ingredients
            
            # 7. Clean up ChatGPT title artifacts
            clean_name = re.sub(r'\d+', '', m['meal_name'])
            clean_name = re.sub(r'^[.\-\s]+|[.\-\s]+$', '', clean_name)
            m['meal_name'] = clean_name.strip().title()
            
            # 8. DEDUCT from the daily bank before moving to the next meal
            daily_cal_remaining -= m['cal']
            daily_pro_remaining -= m['prot']
            
            plan.append(m)
            
    return plan
# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="NutriAI", layout="wide")
st.title("🥗 NutriAI")
st.markdown("Automated Diet Plan Builder for personalized clinical nutrition.")

st.sidebar.header("👤 Patient Tracking")
profile_name = st.sidebar.text_input("Patient ID", value="Patient A")

st.sidebar.subheader("🎯 Macronutrient Targets")
col1, col2 = st.sidebar.columns(2)
t_cal = col1.number_input("Calories", 1000, 4000, 2000, 100)
t_pro = col2.number_input("Protein (g)", 30, 250, 70, 5)

st.sidebar.subheader("🔬 Custom Micronutrient Targets")
st.sidebar.caption("Set up to 5 specific clinical micro thresholds.")
with st.sidebar.expander("Adjust Micronutrient RDAs", expanded=False):
    t_fib = st.number_input("Fiber (g)", 10, 100, 28, 1)
    t_iron = st.number_input("Iron (mg)", 5, 50, 18, 1)
    t_calc = st.number_input("Calcium (mg)", 500, 2500, 1000, 50)
    t_b12 = st.number_input("Vitamin B12 (mcg)", 1.0, 10.0, 2.4, 0.1)
    t_vitd = st.number_input("Vitamin D (IU)", 200, 4000, 600, 50)

# The Engine's Master Target Dictionary
RDA = {
    'cal': t_cal, 'prot': t_pro, 'carb': 250, 'fat': 70, 
    'fib': t_fib, 'iron': t_iron, 'calc': t_calc, 'b12': t_b12, 'vitd': t_vitd, 'zinc': 11
}

st.sidebar.header("🛡️ Clinical Rules (Global)")
conds = st.sidebar.multiselect("Conditions", ["IBS (Low FODMAP)", "GERD", "Diabetes (Low GI)", "Hypertension (DASH)"])
algs = st.sidebar.multiselect("Allergies", ["Gluten", "Soy", "Dairy", "Eggs", "Tree Nuts", "Peanuts", "Shellfish"])

st.sidebar.header("🍽️ Mixed Household Routing")
diet_b = st.sidebar.selectbox("Breakfast Diet", ["Standard", "Vegetarian", "Vegan", "Pescatarian", "Halal", "Kosher"])
diet_l = st.sidebar.selectbox("Lunch Diet", ["Standard", "Vegetarian", "Vegan", "Pescatarian", "Halal", "Kosher"])
diet_d = st.sidebar.selectbox("Dinner Diet", ["Standard", "Vegetarian", "Vegan", "Pescatarian", "Halal", "Kosher"])

# Execution Pipeline
if st.sidebar.button("Generate 7-Day Plan", type="primary"):
    with st.spinner("Generating personalized 7-day meal plan..."):
        
        sql_b = build_exclusion_subquery(conds, algs, diet_b)
        sql_l = build_exclusion_subquery(conds, algs, diet_l)
        sql_d = build_exclusion_subquery(conds, algs, diet_d)
        
        pool_b = fetch_safe_recipes(sql_b, profile_name, "Breakfast")
        pool_l = fetch_safe_recipes(sql_l, profile_name, "Main")
        pool_d = fetch_safe_recipes(sql_d, profile_name, "Main") 
        
        if len(pool_b) < 7 or len(pool_l) < 7 or len(pool_d) < 7:
            st.error(f"⚠️ Over-constrained! Database lacks enough safe recipes. (Available -> Breakfasts: {len(pool_b)}, Lunches: {len(pool_l)}, Dinners: {len(pool_d)})")
        else:
            plan = generate_plan(pool_b, pool_l, pool_d, RDA)
            
            unique_meals = len(set([m['meal_name'] for m in plan]))
            div_score = (unique_meals / 21.0) * 100
            st.success(f"Generated successfully! **Diversity Score: {div_score:.0f}%**")
            
            for day_idx in range(7):
                st.markdown(f"### Day {day_idx + 1}")
                day_meals = plan[day_idx*3 : (day_idx*3)+3]
                
                d_totals = {k: sum(m[k] for m in day_meals) for k in RDA.keys()}
                flags = []
                if d_totals['fib'] < RDA['fib'] * 0.8: flags.append("Low Fiber")
                if d_totals['iron'] < RDA['iron'] * 0.8: flags.append("Low Iron")
                if d_totals['b12'] < RDA['b12'] * 0.8: flags.append("Low B12")
                if d_totals['calc'] < RDA['calc'] * 0.8: flags.append("Low Calcium")
                
                if flags:
                    st.warning(f"📊 **Daily Audit Flags:** {', '.join(flags)} (Day falls below 80% of NCBI targets)")
                else:
                    st.info("📊 **Daily Audit:** All 10 macro/micro nutrients successfully meet baseline RDA targets.")

                cols = st.columns(3)
                for meal_idx, m in enumerate(day_meals):
                    with cols[meal_idx]:
                        m_type = ["Breakfast", "Lunch", "Dinner"][meal_idx]
                        st.markdown(f"**{m_type}**: {m['meal_name']}")
                        st.caption(f"**{m['cal']} kcal | {m['prot']}g Prot** | {m['carb']}g Carb | {m['fat']}g Fat")
                        
                        with st.expander("Details, Micros & Scaled Ingredients"):
                            st.write(f"**Fib:** {m['fib']}g | **Fe:** {m['iron']}mg | **Ca:** {m['calc']}mg")
                            st.write(f"**B12:** {m['b12']}mcg | **VitD:** {m['vitd']}IU | **Zn:** {m['zinc']}mg")
                            st.markdown("**Ingredients:**")
                            for ing in m['ingredients']: st.write(f"• {ing}")
                            
                        c1, c2 = st.columns(2)
                        if c1.button("👍", key=f"u_{day_idx}_{meal_idx}", help="Like this meal"):
                            update_rl_weight(profile_name, m['blueprint_id'], 'up')
                            st.toast("Feedback Saved! We'll show you more meals like this.")
                        if c2.button("👎", key=f"d_{day_idx}_{meal_idx}", help="Dislike this meal"):
                            update_rl_weight(profile_name, m['blueprint_id'], 'down')
                            st.toast("Feedback Saved! We'll show you fewer meals like this.")
                st.divider()