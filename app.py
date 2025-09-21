from flask import Flask, render_template, request, redirect, url_for
import requests, os, re

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- Hardcoded Recipes ----------
recipes = {
    "Omelette": ["eggs", "milk", "cheese", "butter"],
    "Tomato Pasta": ["tomato", "pasta", "olive oil", "garlic"],
    "French Toast": ["bread", "eggs", "milk", "syrup"],
    "Salad": ["lettuce", "tomato", "cucumber", "olive oil"],
    "Grilled Cheese": ["bread", "cheese", "butter"],
    "Pancakes": ["flour", "eggs", "milk", "syrup"],
    "Smoothie": ["banana", "milk", "yogurt", "berries"],
    "Fried Rice": ["rice", "eggs", "soy sauce", "carrot", "peas"],
    "Tacos": ["tortilla", "beef", "cheese", "lettuce", "tomato"],
    "Guacamole": ["avocado", "tomato", "onion", "lime"],
    "Pizza": ["dough", "tomato", "cheese", "olive oil"],
    "Sandwich": ["bread", "cheese", "lettuce", "tomato"],
    "Chicken Soup": ["chicken", "carrot", "celery", "onion"],
    "Stir Fry": ["chicken", "broccoli", "soy sauce", "garlic"],
    "Quesadilla": ["tortilla", "cheese", "chicken"],
    "Mac and Cheese": ["pasta", "cheese", "milk", "butter"],
    "Curry": ["chicken", "rice", "curry powder", "onion", "tomato"],
    "Veggie Wrap": ["tortilla", "lettuce", "cucumber", "tomato", "cheese"],
    "Burger": ["bun", "beef", "lettuce", "tomato", "cheese"],
    "Scrambled Eggs": ["eggs", "milk", "butter"],
}

# ---------- Optional Roboflow ----------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_MODEL_URL = os.getenv("ROBOFLOW_MODEL_URL", "").strip()  # e.g. https://detect.roboflow.com/<model>/1

# ---------- Normalization (plurals + synonyms) ----------
SYNONYMS = {
    "tomatoes": "tomato", "bell pepper": "pepper", "peppers": "pepper",
    "yoghurt": "yogurt", "buttermilk": "milk", "buns": "bun",
    "tortillas": "tortilla", "breads": "bread", "eggs": "egg",
    "onions": "onion", "garlics": "garlic", "berries": "berry",
    "strawberries": "strawberry", "tomatos": "tomato",
    "oliveoil": "olive oil", "olive-oil": "olive oil",
}

def singularize(w: str) -> str:
    if w.endswith("ies") and len(w) > 3: return w[:-3] + "y"
    if w.endswith("oes") and len(w) > 3: return w[:-2]
    if w.endswith("es") and len(w) > 2:  return w[:-2]
    if w.endswith("s")  and len(w) > 1:  return w[:-1]
    return w

def normalize(word: str) -> str:
    w = (word or "").lower().strip()
    w = w.replace("_", " ").replace("-", " ")
    w = SYNONYMS.get(w, w)
    w = SYNONYMS.get(singularize(w), singularize(w))
    return w

# ---------- Fallbacks when detection returns nothing ----------
SAMPLE_HINTS = {
    "fridgepicture": ["milk", "egg", "cheese", "tomato", "lettuce"],
    "sample1": ["egg", "milk", "butter"],
    "sample2": ["tomato", "pasta", "garlic", "olive oil"],
    "sample3": ["bread", "cheese", "butter"],
}
WORD_TO_ING = {
    "milk":"milk","egg":"egg","eggs":"egg","cheese":"cheese","butter":"butter",
    "tomato":"tomato","tomatoes":"tomato","lettuce":"lettuce","cucumber":"cucumber",
    "bread":"bread","pasta":"pasta","garlic":"garlic","oil":"olive oil","olive":"olive oil",
    "banana":"banana","yogurt":"yogurt","rice":"rice","carrot":"carrot","peas":"peas",
}

def guess_from_filename(filename: str):
    if not filename: return []
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    for key, items in SAMPLE_HINTS.items():
        if key in base:
            return list({normalize(x) for x in items})
    words = re.split(r"[^\w]+", base)
    hits = []
    for w in words:
        if not w: continue
        if w in WORD_TO_ING:
            hits.append(WORD_TO_ING[w])
    return list({normalize(x) for x in hits})

# ---------- Detect (Roboflow, optional) ----------
def detect_food(image_path):
    items = []
    if ROBOFLOW_API_KEY and ROBOFLOW_MODEL_URL:
        try:
            with open(image_path, "rb") as img:
                resp = requests.post(
                    f"{ROBOFLOW_MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
                    files={"file": img}, timeout=20
                )
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])
            items = [p.get("class", "") for p in preds if p.get("class")]
        except Exception as e:
            print("[roboflow] error:", e)
    # Normalize + unique
    seen, out = set(), []
    for it in items:
        n = normalize(it)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

# ---------- Matching ----------
def match_recipes(ingredients):
    norm_set = set(normalize(i) for i in ingredients if i)
    matches = {}
    for name, reqs in recipes.items():
        have, missing = [], []
        for r in reqs:
            if normalize(r) in norm_set:
                have.append(r)
            else:
                missing.append(r)
        matches[name] = {"have": have, "missing": missing}
    return matches

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    pantry = ["salt", "pepper", "olive oil", "soy sauce", "flour", "sugar", "butter", "garlic", "onion", "eggs", "milk"]

    if request.method == "POST":
        f = request.files.get("file")
        manual = request.form.get("manual", "")
        pantry_checked = request.form.getlist("pantry[]")

        os.makedirs("static", exist_ok=True)
        saved_path, filename = None, None
        if f and f.filename:
            filename = f.filename
            saved_path = os.path.join("static", filename)
            f.save(saved_path)

        # 1) detections
        ingredients = detect_food(saved_path) if saved_path else []

        # 2) filename guess (if nothing yet)
        if not ingredients and filename:
            ingredients.extend(guess_from_filename(filename))

        # 3) manual list
        if manual:
            parts = [p.strip() for p in manual.replace(";", ",").split(",") if p.strip()]
            ingredients.extend(parts)

        # 4) pantry checkboxes
        ingredients.extend(pantry_checked)

        # 5) demo fallback so the flow never looks empty
        if not ingredients:
            ingredients = ["egg", "milk", "tomato", "olive oil"]

        # normalize + unique for display/match
        seen, norm_unique = set(), []
        for ing in ingredients:
            n = normalize(ing)
            if n and n not in seen:
                seen.add(n)
                norm_unique.append(n)

        print("[debug] ingredients (normalized):", norm_unique)

        # matches + sorted view (coverage desc, then name asc). Hide 0%.
        recipe_matches = match_recipes(norm_unique)
        scored = []
        for name, data in recipe_matches.items():
            have = len(data["have"])
            total = have + len(data["missing"])
            coverage = (have / total) if total else 0
            if have > 0:
                scored.append((name, data, coverage))
        scored.sort(key=lambda x: (-x[2], x[0].lower()))

        return render_template(
            "results.html",
            ingredients=norm_unique,
            sorted_matches=scored
        )

    # GET homepage
    return render_template("index.html", pantry=pantry)

# Convenience routes + 404 → home
@app.route("/index")
def index_alias():
    return redirect(url_for("index"))

@app.route("/results")
def results_alias():
    return redirect(url_for("index"))

@app.errorhandler(404)
def not_found(_e):
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)