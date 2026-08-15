"""Generate deterministic sample resumes and job descriptions for testing.

Produces 100 resumes (mixed .txt/.docx/.pdf) across 12 job families and 6
job descriptions with explicit years requirements and Requirements / Nice to
have blocks. Everything is seeded so downstream tests (test_scenarios.py)
can assert against specific, known candidates.

Run directly: `python generate_sample_data.py`
"""

from __future__ import annotations

import os
import random

from docx import Document
from fpdf import FPDF

RESUME_DIR = "data/resumes"
JD_DIR = "data/job_descriptions"
SEED = 42

FIRST_NAMES = [
    "Avery", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Reese", "Quinn",
    "Skyler", "Rowan", "Elena", "Marcus", "Priya", "Devon", "Sofia", "Nolan",
    "Amara", "Felix", "Junho", "Ines",
]

LAST_NAMES = [
    "Whitfield", "Nakamura", "Okafor", "Bergstrom", "Delacroix", "Huang",
    "Petrova", "Alvarez", "Kowalski", "Singh", "Marsh", "Idowu", "Fontaine",
    "Reyes", "Larsen", "Castellano", "Boateng", "Nilsson", "Kapoor", "Voss",
]

EDUCATION_OPTIONS = [
    "B.S. in Computer Science, University of Michigan",
    "M.S. in Computer Science, Georgia Institute of Technology",
    "B.S. in Information Systems, University of Texas at Austin",
    "M.S. in Data Science, University of Washington",
    "B.A. in Mathematics, UCLA",
    "B.S. in Electrical Engineering, Purdue University",
    "M.B.A., Northwestern University Kellogg School of Management",
    "B.S. in Software Engineering, University of Waterloo",
]

# Job family -> (role titles, core skill pool, section-content blurbs)
JOB_FAMILIES: dict[str, dict[str, object]] = {
    "software_engineer": {
        "titles": ["Software Engineer", "Senior Software Engineer"],
        "skills": ["Python", "Java", "Go", "REST API", "Microservices", "Git",
                   "SQL", "Docker", "Kubernetes", "CI/CD"],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "Senior Data Scientist"],
        "skills": ["Python", "Machine Learning", "scikit-learn", "Pandas",
                   "SQL", "Statistics", "TensorFlow", "PyTorch", "A/B Testing"],
    },
    "product_manager": {
        "titles": ["Product Manager", "Senior Product Manager"],
        "skills": ["Product Management", "Agile", "Scrum", "JIRA",
                   "A/B Testing", "Roadmapping", "Stakeholder Management",
                   "SQL", "Figma"],
    },
    "devops_engineer": {
        "titles": ["DevOps Engineer", "Senior DevOps Engineer"],
        "skills": ["AWS", "Terraform", "Kubernetes", "Docker", "Jenkins",
                   "CI/CD", "Linux", "Bash", "Ansible"],
    },
    "ml_engineer": {
        "titles": ["Machine Learning Engineer", "Senior ML Engineer"],
        "skills": ["Python", "PyTorch", "TensorFlow", "Machine Learning",
                   "Deep Learning", "NLP", "Kubernetes", "MLOps", "AWS"],
    },
    "backend_engineer": {
        "titles": ["Backend Engineer", "Senior Backend Engineer"],
        "skills": ["Java", "Spring", "Python", "Django", "Flask",
                   "PostgreSQL", "Redis", "Kafka", "Microservices", "REST API"],
    },
    "frontend_engineer": {
        "titles": ["Frontend Engineer", "Senior Frontend Engineer"],
        "skills": ["JavaScript", "TypeScript", "React", "Vue", "Angular",
                   "CSS", "HTML", "GraphQL", "Webpack"],
    },
    "data_analyst": {
        "titles": ["Data Analyst", "Senior Data Analyst"],
        "skills": ["SQL", "Excel", "Tableau", "Power BI", "Python",
                   "Data Analysis", "Statistics", "ETL"],
    },
    "cloud_architect": {
        "titles": ["Cloud Architect", "Senior Cloud Architect"],
        "skills": ["AWS", "Azure", "GCP", "Terraform", "Kubernetes",
                   "Microservices", "Networking", "Security", "IAM"],
    },
    "qa_engineer": {
        "titles": ["QA Engineer", "Senior QA Engineer"],
        "skills": ["Selenium", "pytest", "Test Automation", "Java", "Python",
                   "CI/CD", "JIRA", "REST API"],
    },
    "security_engineer": {
        "titles": ["Security Engineer", "Senior Security Engineer"],
        "skills": ["Penetration Testing", "SIEM", "IAM", "OAuth",
                   "Encryption", "Compliance", "GDPR", "Python", "Linux"],
    },
    "mobile_developer": {
        "titles": ["Mobile Developer", "Senior Mobile Developer"],
        "skills": ["Swift", "Kotlin", "iOS", "Android", "Flutter",
                   "React Native", "REST API", "Git"],
    },
}

# Four distinct section-heading phrasings so chunking is exercised.
SECTION_STYLES = [
    {"summary": "Summary", "experience": "Experience", "education": "Education", "skills": "Skills"},
    {"summary": "Objective", "experience": "Work Experience", "education": "Education", "skills": "Technical Skills"},
    {"summary": "Summary", "experience": "Employment", "education": "Education", "skills": "Skills"},
    {"summary": "Objective", "experience": "Experience", "education": "Education", "skills": "Technical Skills"},
]

COMPANIES = [
    "Northlake Systems", "Vantage Digital", "Cobalt Analytics", "Brightwell Labs",
    "Ferry Road Technologies", "Ashgrove Software", "Meridian Cloud", "Ninth & Main",
    "Harborview Data", "Stonebridge Technologies",
]


def _fmt_skills(skills: list[str]) -> str:
    return ", ".join(skills)


def build_resume_text(spec: dict) -> str:
    """Render a resume as plain text using the candidate's section style."""
    style = spec["section_style"]
    name = spec["name"]
    title = spec["title"]
    years = spec["years_experience"]
    skills = spec["skills"]
    education = spec["education"]
    company_a, company_b = spec["companies"]

    lines: list[str] = [name, title, "", style["summary"], ""]
    lines.append(
        f"{title} with {years} years of experience delivering production "
        f"systems using {skills[0]} and {skills[1]}. Proven track record "
        f"across cross-functional teams."
    )
    lines += ["", style["experience"], ""]
    lines.append(f"{title}, {company_a} (Present)")
    lines.append(
        f"- Led development efforts using {skills[2]} and {skills[3]}, "
        f"improving system reliability and delivery speed."
    )
    lines.append(
        f"- Collaborated with stakeholders to ship features leveraging "
        f"{skills[4]}."
    )
    if years >= 4:
        lines.append(f"{title}, {company_b}")
        lines.append(
            f"- Built and maintained services with {skills[1]} and "
            f"{skills[5 % len(skills)]}, supporting {years - 2} years of "
            f"iterative delivery."
        )
    lines += ["", style["education"], ""]
    lines.append(education)
    lines += ["", style["skills"], ""]
    lines.append(_fmt_skills(skills))

    if spec["include_projects"]:
        lines += ["", "Projects", ""]
        lines.append(
            f"- Internal tooling project applying {skills[0]} and "
            f"{skills[-1]} to reduce manual workflow time."
        )
    if spec["include_certifications"]:
        lines += ["", "Certifications", ""]
        lines.append(f"- Certified Associate, {skills[0]} Practitioner Track")

    return "\n".join(lines)


def write_txt(spec: dict, content: str) -> str:
    path = os.path.join(RESUME_DIR, spec["filename"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def write_docx(spec: dict, content: str) -> str:
    path = os.path.join(RESUME_DIR, spec["filename"])
    doc = Document()
    for line in content.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)
    return path


def write_pdf(spec: dict, content: str) -> str:
    path = os.path.join(RESUME_DIR, spec["filename"])
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in content.split("\n"):
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        if not safe_line.strip():
            pdf.ln(6)
        else:
            # Explicit new_x/new_y: multi_cell doesn't reset the cursor to
            # the left margin by default, so the effective width shrinks on
            # every subsequent call until fpdf2 raises "not enough space".
            pdf.multi_cell(0, 6, safe_line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(path)
    return path


def build_candidate_specs(rng: random.Random) -> list[dict]:
    """Build 100 deterministic candidate specs.

    The first 5 are hand-pinned "anchor" candidates with fixed, easily
    assertable attributes (used by test_scenarios.py). The remaining 95 are
    generated from the templates above using the seeded RNG.
    """
    specs: list[dict] = []

    anchors = [
        {
            "name": "Elena Whitfield",
            "family": "ml_engineer",
            "title": "Senior ML Engineer",
            "years_experience": 8,
            "skills": ["Python", "PyTorch", "Machine Learning", "Deep Learning",
                       "NLP", "Kubernetes", "AWS"],
            "education": "M.S. in Data Science, University of Washington",
            "section_style": SECTION_STYLES[0],
            "companies": ("Cobalt Analytics", "Brightwell Labs"),
            "include_projects": True,
            "include_certifications": False,
        },
        {
            "name": "Marcus Nakamura",
            "family": "ml_engineer",
            "title": "Machine Learning Engineer",
            "years_experience": 3,
            "skills": ["Python", "TensorFlow", "Machine Learning", "NLP",
                       "Kubernetes", "MLOps"],
            "education": "B.S. in Computer Science, University of Michigan",
            "section_style": SECTION_STYLES[1],
            "companies": ("Vantage Digital", "Northlake Systems"),
            "include_projects": True,
            "include_certifications": False,
        },
        {
            "name": "Priya Okafor",
            "family": "backend_engineer",
            "title": "Senior Backend Engineer",
            "years_experience": 6,
            "skills": ["Java", "Spring", "PostgreSQL", "Kafka",
                       "Microservices", "REST API"],
            "education": "B.S. in Software Engineering, University of Waterloo",
            "section_style": SECTION_STYLES[2],
            "companies": ("Meridian Cloud", "Ninth & Main"),
            "include_projects": False,
            "include_certifications": True,
        },
        {
            "name": "Devon Bergstrom",
            "family": "frontend_engineer",
            "title": "Frontend Engineer",
            "years_experience": 2,
            "skills": ["JavaScript", "React", "TypeScript", "CSS", "HTML"],
            "education": "B.A. in Mathematics, UCLA",
            "section_style": SECTION_STYLES[3],
            "companies": ("Ashgrove Software", "Harborview Data"),
            "include_projects": True,
            "include_certifications": False,
        },
        {
            "name": "Sofia Delacroix",
            "family": "ml_engineer",
            "title": "Senior ML Engineer",
            "years_experience": 10,
            "skills": ["Python", "PyTorch", "Deep Learning", "Machine Learning",
                       "AWS", "MLOps", "NLP"],
            "education": "M.S. in Computer Science, Georgia Institute of Technology",
            "section_style": SECTION_STYLES[1],
            "companies": ("Stonebridge Technologies", "Cobalt Analytics"),
            "include_projects": False,
            "include_certifications": True,
        },
    ]
    specs.extend(anchors)

    used_names = {(s["name"]) for s in anchors}
    families = list(JOB_FAMILIES.keys())
    name_pool = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(name_pool)
    name_iter = iter(name_pool)

    while len(specs) < 100:
        first, last = next(name_iter)
        name = f"{first} {last}"
        if name in used_names:
            continue
        used_names.add(name)

        family_key = families[len(specs) % len(families)]
        family = JOB_FAMILIES[family_key]
        title = rng.choice(family["titles"])
        years = rng.randint(1, 18)
        pool = family["skills"]
        k = rng.randint(min(5, len(pool)), len(pool))
        skills = rng.sample(pool, k)
        education = rng.choice(EDUCATION_OPTIONS)
        style = SECTION_STYLES[len(specs) % len(SECTION_STYLES)]
        companies = tuple(rng.sample(COMPANIES, 2))

        specs.append(
            {
                "name": name,
                "family": family_key,
                "title": title,
                "years_experience": years,
                "skills": skills,
                "education": education,
                "section_style": style,
                "companies": companies,
                "include_projects": rng.random() < 0.4,
                "include_certifications": rng.random() < 0.3,
            }
        )

    # Assign formats: 35 txt, 35 docx, 30 pdf, deterministically shuffled.
    formats = ["txt"] * 35 + ["docx"] * 35 + ["pdf"] * 30
    rng.shuffle(formats)
    for spec, fmt in zip(specs, formats):
        slug = spec["name"].lower().replace(" ", "_")
        spec["format"] = fmt
        spec["filename"] = f"{slug}_{spec['family']}.{fmt}"

    return specs


def generate_resumes() -> list[dict]:
    os.makedirs(RESUME_DIR, exist_ok=True)
    rng = random.Random(SEED)
    specs = build_candidate_specs(rng)

    writers = {"txt": write_txt, "docx": write_docx, "pdf": write_pdf}
    for spec in specs:
        content = build_resume_text(spec)
        writers[spec["format"]](spec, content)

    return specs


JOB_DESCRIPTIONS = [
    {
        "filename": "senior_ml_engineer.txt",
        "content": """Senior ML Engineer

We are hiring a Senior ML Engineer to build and deploy production machine
learning systems for our recommendation platform.

Requirements:
- 5+ years of experience in machine learning engineering
- Strong Python and PyTorch experience
- Experience with Deep Learning and NLP
- Experience deploying models with Kubernetes

Nice to have:
- Experience with MLOps tooling
- AWS experience
- Experience mentoring junior engineers
""",
    },
    {
        "filename": "backend_software_engineer.txt",
        "content": """Backend Software Engineer

Join our platform team building the services that power our core product.

Requirements:
- 3+ years of backend engineering experience
- Proficiency in Java and Spring
- Experience with PostgreSQL and Microservices
- Experience with REST API design

Nice to have:
- Kafka experience
- Experience with Docker and CI/CD
""",
    },
    {
        "filename": "cloud_devops_engineer.txt",
        "content": """Cloud DevOps Engineer

We need a DevOps Engineer to own our cloud infrastructure and deployment
pipelines.

Requirements:
- 4+ years of DevOps or infrastructure engineering experience
- Hands-on AWS experience
- Terraform and Kubernetes experience
- Experience with CI/CD pipelines (Jenkins or similar)

Nice to have:
- Ansible experience
- Bash scripting
""",
    },
    {
        "filename": "product_manager.txt",
        "content": """Product Manager

We're looking for a Product Manager to own the roadmap for our analytics
product line.

Requirements:
- 4+ years of product management experience
- Experience running A/B tests
- Strong stakeholder management skills
- Experience with Agile/Scrum practices

Nice to have:
- SQL proficiency
- Experience with Figma
""",
    },
    {
        "filename": "data_analyst.txt",
        "content": """Data Analyst

We're hiring a Data Analyst to support decision-making across the business
with dashboards and ad hoc analysis.

Requirements:
- 2+ years of data analysis experience
- Strong SQL skills
- Experience with Tableau or Power BI
- Experience with Excel

Nice to have:
- Python experience
- Experience with ETL pipelines
""",
    },
    {
        "filename": "security_engineer.txt",
        "content": """Security Engineer

We are looking for a Security Engineer to strengthen our application and
infrastructure security posture.

Requirements:
- 5+ years of security engineering experience
- Experience with penetration testing
- Experience with IAM and OAuth
- Familiarity with compliance frameworks (GDPR or similar)

Nice to have:
- SIEM tooling experience
- Python scripting experience
""",
    },
]


def generate_job_descriptions() -> None:
    os.makedirs(JD_DIR, exist_ok=True)
    for jd in JOB_DESCRIPTIONS:
        path = os.path.join(JD_DIR, jd["filename"])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(jd["content"])


def main() -> None:
    specs = generate_resumes()
    generate_job_descriptions()

    by_format = {"txt": 0, "docx": 0, "pdf": 0}
    for spec in specs:
        by_format[spec["format"]] += 1

    print(f"Generated {len(specs)} resumes in {RESUME_DIR}/")
    print(f"  formats: {by_format}")
    print(f"  job families: {len(JOB_FAMILIES)}")
    print(f"Generated {len(JOB_DESCRIPTIONS)} job descriptions in {JD_DIR}/")


if __name__ == "__main__":
    main()
