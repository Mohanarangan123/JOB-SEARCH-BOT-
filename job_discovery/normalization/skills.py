"""
Skills normalizer — static taxonomy only.

Maps aliases, abbreviations, and variant spellings to canonical skill names.

Rules:
  - Static lookup table only — no embeddings, no LLM, no fuzzy matching.
  - Unmapped skills are preserved as-is (cleaned).
  - Preserves original alias as evidence.
  - Never invents or removes skills.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class NormalizedSkill:
    canonical: str
    original:  str
    alias:     Optional[str] = None   # alias used in input if different from canonical


# ─────────────────────────────────────────────────────────────────────────────
# Static taxonomy  (alias → canonical)
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY: Dict[str, str] = {
    # Machine Learning / AI
    "ml":                   "Machine Learning",
    "machine learning":     "Machine Learning",
    "machine-learning":     "Machine Learning",
    "deep learning":        "Deep Learning",
    "dl":                   "Deep Learning",
    "deep-learning":        "Deep Learning",
    "natural language processing": "NLP",
    "nlp":                  "NLP",
    "computer vision":      "Computer Vision",
    "cv":                   "Computer Vision",
    "ai":                   "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "llm":                  "LLM",
    "large language model": "LLM",
    "generative ai":        "Generative AI",
    "gen ai":               "Generative AI",

    # Languages
    "python":               "Python",
    "python3":              "Python",
    "java":                 "Java",
    "javascript":           "JavaScript",
    "js":                   "JavaScript",
    "typescript":           "TypeScript",
    "ts":                   "TypeScript",
    "golang":               "Go",
    "go":                   "Go",
    "rust":                 "Rust",
    "c++":                  "C++",
    "c plus plus":          "C++",
    "cplusplus":            "C++",
    "c#":                   "C#",
    "csharp":               "C#",
    "dotnet":               ".NET",
    ".net":                 ".NET",
    "ruby":                 "Ruby",
    "php":                  "PHP",
    "scala":                "Scala",
    "kotlin":               "Kotlin",
    "swift":                "Swift",
    "r":                    "R",

    # Web Frameworks
    "react":                "React",
    "reactjs":              "React",
    "react.js":             "React",
    "angular":              "Angular",
    "angularjs":            "Angular",
    "vue":                  "Vue.js",
    "vuejs":                "Vue.js",
    "vue.js":               "Vue.js",
    "node":                 "Node.js",
    "nodejs":               "Node.js",
    "node.js":              "Node.js",
    "django":               "Django",
    "flask":                "Flask",
    "fastapi":              "FastAPI",
    "spring":               "Spring",
    "spring boot":          "Spring Boot",
    "springboot":           "Spring Boot",
    "rails":                "Ruby on Rails",
    "ruby on rails":        "Ruby on Rails",
    "express":              "Express.js",
    "expressjs":            "Express.js",
    "laravel":              "Laravel",
    "nextjs":               "Next.js",
    "next.js":              "Next.js",

    # Databases
    "postgres":             "PostgreSQL",
    "postgresql":           "PostgreSQL",
    "mysql":                "MySQL",
    "mongodb":              "MongoDB",
    "mongo":                "MongoDB",
    "redis":                "Redis",
    "elasticsearch":        "Elasticsearch",
    "elastic search":       "Elasticsearch",
    "cassandra":            "Cassandra",
    "sqlite":               "SQLite",
    "mssql":                "SQL Server",
    "sql server":           "SQL Server",
    "oracle":               "Oracle DB",
    "oracle db":            "Oracle DB",
    "dynamodb":             "DynamoDB",
    "neo4j":                "Neo4j",
    "clickhouse":           "ClickHouse",

    # Cloud
    "aws":                  "AWS",
    "amazon web services":  "AWS",
    "azure":                "Azure",
    "microsoft azure":      "Azure",
    "gcp":                  "GCP",
    "google cloud":         "GCP",
    "google cloud platform":"GCP",

    # DevOps / Infra
    "docker":               "Docker",
    "kubernetes":           "Kubernetes",
    "k8s":                  "Kubernetes",
    "terraform":            "Terraform",
    "ansible":              "Ansible",
    "jenkins":              "Jenkins",
    "github actions":       "GitHub Actions",
    "ci/cd":                "CI/CD",
    "cicd":                 "CI/CD",
    "devops":               "DevOps",
    "sre":                  "SRE",
    "linux":                "Linux",
    "unix":                 "Unix",
    "bash":                 "Bash",
    "shell scripting":      "Shell Scripting",
    "git":                  "Git",

    # Data
    "sql":                  "SQL",
    "spark":                "Apache Spark",
    "apache spark":         "Apache Spark",
    "hadoop":               "Hadoop",
    "kafka":                "Apache Kafka",
    "apache kafka":         "Apache Kafka",
    "airflow":              "Apache Airflow",
    "apache airflow":       "Apache Airflow",
    "pandas":               "Pandas",
    "numpy":                "NumPy",
    "scikit-learn":         "scikit-learn",
    "sklearn":              "scikit-learn",
    "tensorflow":           "TensorFlow",
    "tf":                   "TensorFlow",
    "pytorch":              "PyTorch",
    "torch":                "PyTorch",
    "huggingface":          "Hugging Face",
    "hugging face":         "Hugging Face",
    "powerbi":              "Power BI",
    "power bi":             "Power BI",
    "tableau":              "Tableau",
    "looker":               "Looker",

    # Practices
    "agile":                "Agile",
    "scrum":                "Scrum",
    "rest":                 "REST",
    "restful":              "REST",
    "rest api":             "REST",
    "graphql":              "GraphQL",
    "microservices":        "Microservices",
    "tdd":                  "TDD",
    "test driven development": "TDD",
    "bdd":                  "BDD",
    "oop":                  "OOP",
    "object oriented programming": "OOP",
    "object-oriented":      "OOP",
}


def _clean_skill(raw: str) -> str:
    """Strip punctuation (except / # + . -), collapse whitespace, lowercase for lookup."""
    cleaned = raw.strip().lower()
    # Keep / # + . # - but remove other punctuation
    cleaned = re.sub(r"[^\w\s.#+\-/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class SkillsNormalizer:
    """
    Normalize a list of skill strings using the static taxonomy.
    """

    def normalize(self, skills: List[str]) -> List[NormalizedSkill]:
        """
        Normalize a list of raw skill strings.

        Returns a deduplicated list of NormalizedSkill objects.
        Unmapped skills are included as-is (cleaned).
        """
        result: List[NormalizedSkill] = []
        seen_canonicals: Set[str] = set()

        for raw in skills:
            if not raw or not raw.strip():
                continue
            clean = _clean_skill(raw)
            canonical = _TAXONOMY.get(clean)

            if canonical is None:
                # Preserve as-is but with cleaned presentation
                canonical = raw.strip()

            if canonical.lower() not in seen_canonicals:
                seen_canonicals.add(canonical.lower())
                result.append(NormalizedSkill(
                    canonical=canonical,
                    original=raw.strip(),
                    alias=raw.strip() if raw.strip().lower() != canonical.lower() else None,
                ))

        return result

    def normalize_single(self, raw: str) -> NormalizedSkill:
        """Normalize one skill string."""
        results = self.normalize([raw])
        return results[0] if results else NormalizedSkill(canonical=raw, original=raw)
