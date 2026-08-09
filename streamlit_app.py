from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

API_BASE_URL = os.getenv("JOB_DISCOVERY_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Job Discovery V1", page_icon="📄", layout="wide")
st.title("Job Discovery System")
st.caption("V1 disposition: query, discover, rank, and export job listings")


@st.cache_data(show_spinner=False, ttl=30)
def fetch_jobs() -> list[Dict[str, Any]]:
    url = f"{API_BASE_URL}/api/jobs"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Unable to load jobs from API: {exc}")
        return []


@st.cache_data(show_spinner=False, ttl=10)
def fetch_health() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


with st.sidebar:
    st.header("API")
    st.text_input("API Base URL", value=API_BASE_URL, key="api_base_url")

    st.header("Search Criteria")
    title = st.text_input("Title", value="Python Developer")
    location = st.text_input("Location", value="Chennai")
    employment_type = st.selectbox(
        "Employment Type",
        ["", "full-time", "part-time", "contract", "internship"],
        index=0,
    )
    workplace_type = st.selectbox(
        "Workplace Type",
        ["", "remote", "hybrid", "onsite"],
        index=0,
    )
    experience_years_min = st.number_input("Experience min", min_value=0, max_value=30, value=0)
    experience_years_max = st.number_input("Experience max", min_value=0, max_value=30, value=2)
    preferred_sources = st.multiselect(
        "Sources",
        ["linkedin", "indeed", "naukri", "cutshort", "instahyre", "hirist", "wellfound", "hirect"],
        default=["naukri", "indeed", "linkedin"],
    )

    if st.button("Trigger Search"):
        payload = {
            "title": title or None,
            "keywords": [title] if title else None,
            "location": location or None,
            "employment_type": employment_type or None,
            "workplace_type": workplace_type or None,
            "experience_years_min": int(experience_years_min) if experience_years_min is not None else None,
            "experience_years_max": int(experience_years_max) if experience_years_max is not None else None,
            "preferred_sources": preferred_sources or None,
        }
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/api/jobs/search",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            st.success(f"Search created: {data.get('run_id')}")
        except Exception as exc:
            st.error(f"Search request failed: {exc}")

    st.header("Export")
    export_source = st.text_input("Source", value="")
    export_location = st.text_input("Location filter", value="")
    export_posted_within = st.number_input("Posted within days", min_value=0, max_value=365, value=0)
    export_exp_min = st.number_input("Exp min for export", min_value=0, max_value=30, value=0)
    export_exp_max = st.number_input("Exp max for export", min_value=0, max_value=30, value=10)

    if st.button("Export XLSX"):
        params = {}
        if export_source:
            params["source"] = export_source
        if export_location:
            params["location"] = export_location
        if export_posted_within:
            params["posted_within_days"] = int(export_posted_within)
        if export_exp_min:
            params["experience_years_min"] = int(export_exp_min)
        if export_exp_max:
            params["experience_years_max"] = int(export_exp_max)

        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/api/jobs/export",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            st.success(payload.get("message"))
        except Exception as exc:
            st.error(f"Export failed: {exc}")


status = fetch_health()
if status:
    st.success("Backend status: connected")
else:
    st.warning("Backend status: not reachable at the configured API URL")


jobs = fetch_jobs()
if jobs:
    st.subheader("Job listings")
    jobs_to_show = []
    for job in jobs:
        details = job.get("details") or {}
        company = (details.get("company") or {}).get("name")
        jobs_to_show.append(
            {
                "Title": details.get("title"),
                "Company": company,
                "Location": details.get("location"),
                "Employment Type": details.get("employment_type"),
                "Workplace": details.get("work_mode"),
                "Source": (job.get("source") or {}).get("source_name"),
                "Job URL": job.get("canonical_url"),
                "Validation": (job.get("validation") or {}).get("status"),
                "Lifecycle": job.get("lifecycle_status"),
            }
        )
    st.dataframe(jobs_to_show, use_container_width=True)
else:
    st.info("No jobs returned from API yet")
