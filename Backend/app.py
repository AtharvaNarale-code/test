import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from Backend.Extraction import (
    pdf_extract,
    list_to_json,
    extract_skill_from_resume,
    generate_analysis_metrics,
    calculate_skill_strength,
)
import Backend.Skilldomain as Skilldomain
from Backend.llm import get_recruiter_note, get_candidate_roadmap
from Backend.Ranking import get_domain_leaderboard, build_leaderboard_summary

load_dotenv()   
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_KEY_1=os.getenv("GEMINI_API_KEY_1")  

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB cap


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page — choose HR or Candidate view."""
    return render_template("index.html")


@app.route("/hr")
def hr_view():
    """Full HR dashboard."""
    return render_template("hr.html")




# ---------------------------------------------------------------------------
# Core pipeline helper
# ---------------------------------------------------------------------------

def _process_single_resume(file, candidate_name: str, domain: str) -> dict:
    """
    PDF → Extraction → Scoring → AI Recruiter Note.
    Always returns a dict; errors are captured, never raised.
    """
    try:
        # 1. Extract text from PDF
        raw_text = pdf_extract(file)
        if not raw_text:
            raise ValueError("Empty PDF — could not extract text.")

        # 2. Section parsing + skill extraction
        structured = list_to_json(raw_text)
        skills     = extract_skill_from_resume(structured, Skilldomain.SKILL_DICT)
        metrics    = generate_analysis_metrics(skills)

        # 3. Weighted scoring (strong=1.5, medium=1.0, weak=0.5)
        scoring = calculate_skill_strength(metrics)
        score   = scoring["skill_strength_score"]

        # 4. LLM recruiter note 
        note = get_recruiter_note(GEMINI_KEY, candidate_name, domain, skills, score)
        return {
            "name":           candidate_name,
            "score":          score,
            "weighted_sum":   scoring["weighted_sum"],
            "net_skills":     scoring["net_skills"],
            "metrics":        metrics,
            "skills":         skills,
            "recruiter_note": note,
        }

    except Exception as exc:
        return {
            "name":           candidate_name,
            "score":          0.0,
            "weighted_sum":   0.0,
            "net_skills":     0,
            "metrics":        {},
            "skills":         {},
            "recruiter_note": f"Processing failed: {exc}",
            "error":          str(exc),
        }

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/leaderboard", methods=["POST"])
def leaderboard():
   
    files  = request.files.getlist("resumes")
    names  = request.form.getlist("names")   # optional; may be partial
    domain = request.form.get("domain", "Software Engineering").strip()

    # Validation
    valid_files = [f for f in files if f and f.filename != ""]
    if not valid_files:
        return jsonify({"error": "No PDF files provided."}), 400

    # Process every resume
    candidates = []
    for i, file in enumerate(valid_files):
        # Derive name: use provided value or fall back to filename stem
        if i < len(names) and names[i].strip():
            name = names[i].strip()
        else:
            stem = os.path.splitext(file.filename)[0]
            name = stem.replace("_", " ").replace("-", " ").title()

        candidates.append(_process_single_resume(file, name, domain))

    # Rank and summarise
    ranked  = get_domain_leaderboard(candidates)
    summary = build_leaderboard_summary(ranked)

    return jsonify({
        "status":           "success",
        "domain":           domain,
        "total_candidates": len(ranked),
        "summary":          summary,
        "leaderboard":      ranked,
    })


@app.route("/api/recruiter-note", methods=["POST"])
def recruiter_note_endpoint():
    """
    Regenerate a recruiter note for one candidate on demand.

    Expects JSON body:
        { "name": str, "domain": str, "skills": dict, "score": float }

    Returns:
        { "recruiter_note": str }
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Missing or invalid JSON body."}), 400

    note = get_recruiter_note(
        candidate_name  = body.get("name",   "Candidate"),
        domain          = body.get("domain", "Software Engineering"),
        extracted_skills= body.get("skills", {}),
        score           = float(body.get("score", 0.0)),
    )
    return jsonify({"recruiter_note": note})

@app.route("/candidate")
def candidate_view(): return render_template("candidate.html")

@app.route("/suggest")
def suggest_view(): return render_template("suggest.html")

# --- CANDIDATE API ---
@app.route("/api/candidate/analyse", methods=["POST"])
def candidate_analyse():
    """Single resume analysis for the Candidate Portal."""
    file = request.files.get("resume")
    name = request.form.get("name", "Candidate")
    domain = request.form.get("domain", "Software Engineering")
    
    # standard extraction pipeline
    raw_text = pdf_extract(file)
    structured = list_to_json(raw_text)
    skills = extract_skill_from_resume(structured, Skilldomain.SKILL_DICT)
    metrics = generate_analysis_metrics(skills)
    scoring = calculate_skill_strength(metrics)
    
    # Get the HR note (used as a summary on the result page)
    note = get_recruiter_note(GEMINI_KEY, name, domain, skills, scoring["skill_strength_score"])
    
    return jsonify({
        "name": name,
        "domain": domain,
        "score": scoring["skill_strength_score"],
        "metrics": metrics,
        "skills": skills,
        "recruiter_note": note
    })

@app.route("/api/candidate/suggest", methods=["POST"])
def candidate_suggest():
    """Generates the AI roadmap and Mermaid chart."""
    data = request.json
    ai_roadmap = get_candidate_roadmap(
        GEMINI_KEY, data['name'], data['domain'], data['skills'], data['score']
    )
    return jsonify(ai_roadmap)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
