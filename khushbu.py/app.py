import os
# Configure Matplotlib to use a writable temp directory and non-interactive backend before any other imports
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file, session
import json
import io
import zipfile
import pandas as pd
import requests

# Modern Google GenAI SDK namespace import configuration
try:
    from google import genai
except ImportError:
    genai = None

# Optional internal PDF reading handling
try:
    import fitz
except ImportError:
    fitz = None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-academic-key-123987")

# Vercel Pro configuration via global declaration for deployment awareness
# Note: Vercel Hobby tier times out at 10 seconds. Pro accounts can utilize up to 60 seconds.
maxDuration = 60

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)

def call_gemini_flash_lite(prompt: str) -> str:
    client = get_gemini_client()
    if not client:
        return "ERROR: No Gemini API key configured in environment variables or SDK missing."
    try:
        # Utilizing exact model specified via modern client implementation
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"

def extract_pdf_metadata(file_bytes, filename) -> dict:
    if fitz is None:
        return {"title": filename, "authors": "Unknown", "abstract": "", "raw_text": "PyMuPDF not installed", "filename": filename}
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        meta = doc.metadata or {}
        title = meta.get("title", "") or filename.replace(".pdf", "")
        authors = meta.get("author", "") or "Unknown"
        first_page_text = ""
        if len(doc) > 0:
            first_page_text = doc[0].get_text()[:1500]
        abstract = ""
        lower = first_page_text.lower()
        if "abstract" in lower:
            idx = lower.index("abstract")
            abstract = first_page_text[idx:idx+800].strip()
        doc.close()
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "raw_text": first_page_text,
            "filename": filename
        }
    except Exception as e:
        return {"title": filename, "authors": "Unknown", "abstract": "", "raw_text": str(e), "filename": filename}

def generate_bibtex(references: list) -> str:
    entries = []
    for i, ref in enumerate(references):
        key = f"ref{i+1}"
        title = ref.get("title", f"Reference {i+1}").replace("{", "").replace("}", "")
        authors = ref.get("authors", "Unknown Author")
        entry = f"""@article{{{key},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{2024}},
  journal = {{Unknown Journal}}
}}"""
        entries.append(entry)
    return "\n\n".join(entries)

def build_latex(project, authors, sections, bibtex_content) -> str:
    author_str = " \\and ".join([a.get("name", "Author") for a in authors]) if authors else "Author Name"
    affil = authors[0].get("affiliation", "University") if authors else "University"

    def safe(text):
        if not text:
            return "This section has not been generated yet."
        return text.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_").replace("^", "\\^{}")

    latex = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{natbib}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\usepackage{{booktabs}}
\\usepackage{{setspace}}
\\geometry{{margin=1in}}
\\onehalfspacing

\\title{{{safe(project.get('title', 'Research Paper'))}}}
\\author{{{safe(author_str)} \\\\ \\small {safe(affil)}}}
\\date{{\\today}}

\\bibliographystyle{{plain}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{safe(sections.get('abstract', ''))}
\\end{{abstract}}

\\textbf{{Keywords:}} {safe(project.get('keywords', ''))}

\\newpage
\\tableofcontents
\\newpage

\\section{{Introduction}}
{safe(sections.get('introduction', ''))}

\\section{{Literature Review}}
{safe(sections.get('literature_review', ''))}

\\section{{Methodology}}
{safe(sections.get('methodology', ''))}

\\section{{Results}}
{safe(sections.get('results', ''))}

\\section{{Discussion}}
{safe(sections.get('discussion', ''))}

\\section{{Conclusion}}
{safe(sections.get('conclusion', ''))}

\\bibliography{{references}}

\\end{{document}}
"""
    return latex

def run_integrity_check(project, authors, references, planner, sections) -> dict:
    return {
        "Research Title": bool(project.get("title", "").strip()),
        "Authors Added": len(authors) > 0,
        "References Added": len(references) > 0,
        "Research Objective": bool(planner.get("objective", "")),
        "Abstract Generated": bool(sections.get("abstract", "").strip()),
        "Introduction Generated": bool(sections.get("introduction", "").strip()),
        "Literature Review Generated": bool(sections.get("literature_review", "").strip()),
        "Methodology Generated": bool(sections.get("methodology", "").strip()),
        "Conclusion Generated": bool(sections.get("conclusion", "").strip()),
    }

def init_session_state():
    if "project" not in session:
        session["project"] = {
            "title": "", "domain": "", "research_type": "Review Paper",
            "citation_style": "IEEE", "journal_template": "Generic Academic",
            "num_pages": 8, "keywords": ""
        }
    if "authors" not in session:
        session["authors"] = []
    if "references" not in session:
        session["references"] = []
    if "bibtex" not in session:
        session["bibtex"] = ""
    if "planner" not in session:
        session["planner"] = {}
    if "sections" not in session:
        session["sections"] = {
            "abstract": "", "introduction": "", "literature_review": "",
            "methodology": "", "results": "", "discussion": "", "conclusion": ""
        }
    if "figures" not in session:
        session["figures"] = []
    if "charts" not in session:
        session["charts"] = []

@app.route("/", methods=["GET"])
def index():
    init_session_state()
    return render_template("index.html")

@app.route("/api/state", methods=["GET", "POST"])
def manage_state():
    init_session_state()
    if request.method == "POST":
        data = request.get_json() or {}
        if "project" in data: session["project"] = data["project"]
        if "sections" in data: session["sections"] = data["sections"]
        session.modified = True
        return jsonify({"status": "success"})
    
    checks = run_integrity_check(
        session["project"], session["authors"], session["references"],
        session["planner"], session["sections"]
    )
    return jsonify({
        "project": session["project"],
        "authors": session["authors"],
        "references": session["references"],
        "bibtex": session["bibtex"],
        "planner": session["planner"],
        "sections": session["sections"],
        "figures": [{"name": f["name"]} for f in session["figures"]],
        "charts": [{"name": c["name"]} for c in session["charts"]],
        "integrity_checks": checks
    })

@app.route("/api/authors/add", methods=["POST"])
def add_author():
    init_session_state()
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if name:
        session["authors"].append({
            "name": name,
            "affiliation": data.get("affiliation", ""),
            "department": data.get("department", ""),
            "email": data.get("email", "")
        })
        session.modified = True
        return jsonify({"status": "success", "authors": session["authors"]})
    return jsonify({"status": "error", "message": "Author name required"}), 400

@app.route("/api/authors/remove", methods=["POST"])
def remove_author():
    init_session_state()
    data = request.get_json() or {}
    idx = data.get("index", -1)
    if 0 <= idx < len(session["authors"]):
        session["authors"].pop(idx)
        session.modified = True
        return jsonify({"status": "success", "authors": session["authors"]})
    return jsonify({"status": "error", "message": "Invalid index"}), 400

@app.route("/api/references/upload", methods=["POST"])
def upload_reference():
    init_session_state()
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(".pdf"):
        file_bytes = file.read()
        existing_names = {r.get("filename", "") for r in session["references"]}
        if file.filename not in existing_names:
            meta = extract_pdf_metadata(file_bytes, file.filename)
            session["references"].append(meta)
            session["bibtex"] = generate_bibtex(session["references"])
            session.modified = True
        return jsonify({"status": "success", "references": session["references"], "bibtex": session["bibtex"]})
    return jsonify({"status": "error", "message": "Invalid file type"}), 400

@app.route("/api/planner/generate", methods=["POST"])
def generate_plan():
    init_session_state()
    p = session["project"]
    if not p.get("title"):
        return jsonify({"status": "error", "message": "Please complete Project Setup first."}), 400
    
    prompt = f"""You are an academic research assistant helping a first-year college student.

Research title: {p['title']}
Domain: {p['domain']}
Type: {p['research_type']}
Keywords: {p['keywords']}

Generate a research plan as valid JSON with these exact keys:
- objective: one clear sentence
- research_questions: list of 3-5 strings
- keywords: list of 6-10 strings
- structure: list of section names with one-sentence descriptions

Respond ONLY with valid JSON. No markdown, no explanation."""

    result = call_gemini_flash_lite(prompt)
    if result.startswith("ERROR:"):
        return jsonify({"status": "error", "message": result}), 500
    
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        session["planner"] = parsed
        session.modified = True
        return jsonify({"status": "success", "planner": parsed})
    except Exception:
        session["planner"] = {"raw": result}
        session.modified = True
        return jsonify({"status": "raw", "planner": session["planner"]})

@app.route("/api/sections/generate", methods=["POST"])
def generate_section():
    init_session_state()
    data = request.get_json() or {}
    key = data.get("key", "")
    
    p = session["project"]
    planner = session["planner"]
    objective = planner.get("objective", p.get("title", ""))
    refs_context = "; ".join([r.get("title", "") for r in session["references"][:5]])
    
    section_configs = {
        "abstract": f"Write a 200-word academic abstract for a {p['research_type']} titled '{p['title']}' in {p['domain']}. Research objective: {objective}. Keywords: {p.get('keywords','')}. Use formal academic English. No fake citations.",
        "introduction": f"Write an introduction section (300-400 words) for a {p['research_type']} titled '{p['title']}' in {p['domain']}. Objective: {objective}. Include: background, problem statement, paper organization. Academic tone. No fabricated statistics.",
        "literature_review": f"Write a literature review section (400-500 words) for '{p['title']}'. Domain: {p['domain']}. Referenced papers: {refs_context if refs_context else 'none provided'}. Discuss research themes without fabricating specific paper details or fake citations. Academic tone.",
        "methodology": f"Write a methodology section (300-400 words) for a {p['research_type']} titled '{p['title']}'. Research type: {p['research_type']}. Objective: {objective}. Describe research approach, data collection concept, and analysis method. No fabricated experiments.",
        "results": f"Create a results section template (250-300 words) for '{p['title']}'. Include placeholder structure, table headings, and figure references that the student can fill in. Label placeholders clearly with [INSERT ...].",
        "discussion": f"Write a discussion section (300-400 words) for '{p['title']}'. Objective: {objective}. Discuss implications, limitations, and future work conceptually. Academic tone. No fabricated data.",
        "conclusion": f"Write a conclusion section (200-250 words) for '{p['title']}'. Objective: {objective}. Summarize key findings conceptually and state research contributions. Academic tone."
    }
    
    if key not in section_configs:
        return jsonify({"status": "error", "message": "Invalid section key"}), 400
        
    result = call_gemini_flash_lite(section_configs[key])
    if result.startswith("ERROR:"):
        return jsonify({"status": "error", "message": result}), 500
        
    session["sections"][key] = result
    session.modified = True
    return jsonify({"status": "success", "content": result})

@app.route("/api/figures/upload", methods=["POST"])
def upload_figure():
    init_session_state()
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400
        
    data = file.read()
    if not any(f["name"] == file.filename for f in session["figures"]):
        session["figures"].append({"name": file.filename, "data": data.hex()})
        session.modified = True
    return jsonify({"status": "success", "figures": [{"name": f["name"]} for f in session["figures"]]})

@app.route("/api/charts/csv", methods=["POST"])
def generate_chart_from_csv():
    init_session_state()
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    file = request.files["file"]
    chart_type = request.form.get("chart_type", "Bar Chart")
    x_col = request.form.get("x_col", "")
    y_col = request.form.get("y_col", "")
    chart_title = request.form.get("chart_title", "Research Data Chart")
    
    try:
        df = pd.read_csv(file)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#ffffff")

        if chart_type == "Bar Chart":
            ax.bar(df[x_col].astype(str), df[y_col], color="#1e40af", edgecolor="white", linewidth=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "Line Chart":
            ax.plot(df[x_col].astype(str), df[y_col], color="#1e40af", linewidth=2, marker="o", markersize=5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "Pie Chart":
            ax.pie(df[y_col], labels=df[x_col].astype(str), autopct="%1.1f%%", colors=plt.cm.Blues(range(50, 250, 200 // max(len(df), 1))))
            ax.axis("equal")

        ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        img_buf.seek(0)
        chart_data = img_buf.read()
        plt.close()

        session["charts"].append({"name": chart_title, "data": chart_data.hex()})
        session.modified = True
        return jsonify({"status": "success", "chart_name": chart_title})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/export/latex", methods=["GET"])
def preview_latex():
    init_session_state()
    latex_content = build_latex(session["project"], session["authors"], session["sections"], session["bibtex"])
    return latex_content, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route("/export/zip", methods=["GET"])
def download_zip():
    init_session_state()
    latex_content = build_latex(session["project"], session["authors"], session["sections"], session["bibtex"])
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project/main.tex", latex_content)
        zf.writestr("project/references.bib", session.get("bibtex") or "% No references added")
        zf.writestr("project/images/.gitkeep", "")
        zf.writestr("project/charts/.gitkeep", "")
        
        for i, fig in enumerate(session.get("figures", [])):
            try:
                zf.writestr(f"project/images/figure_{i+1}.png", bytes.fromhex(fig["data"]))
            except Exception:
                pass
        for i, chart in enumerate(session.get("charts", [])):
            try:
                zf.writestr(f"project/charts/chart_{i+1}.png", bytes.fromhex(chart["data"]))
            except Exception:
                pass
                
    buf.seek(0)
    project_name = session["project"].get("title", "research_paper").replace(" ", "_")[:40]
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{project_name}_latex.zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)