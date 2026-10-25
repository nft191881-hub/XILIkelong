import re
from typing import List, Dict, Any

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="XILIkelong Stack Inference API")

FRONTEND_RULES = {
    "React": ["react", "jsx", "useState", "useEffect", "createRoot", "next.js", "nextjs"],
    "Vue": ["vue", "v-model", "defineComponent", "nuxt", "vue-router"],
    "Angular": ["@angular", "ngmodule", "selector:", "angular", "ng-app"],
    "Svelte": ["svelte", "onMount", "$store", "sveltekit"],
    "jQuery": ["jquery", "$(", "document.ready"],
}

BACKEND_RULES = {
    "FastAPI": ["fastapi", "APIRouter", "@app.get", "uvicorn", "pydantic"],
    "Django": ["django", "urlpatterns", "models.Model", "manage.py", "settings.py"],
    "Flask": ["flask", "app.route", "render_template", "jinja2"],
    "Express": ["express", "require('express')", "app.use", "router.get", "axios"],
    "Spring Boot": ["springframework", "@RestController", "@Controller", "pom.xml", "application.properties"],
    "ASP.NET": ["asp.net", "WebApplication", "IActionResult", "Startup.cs"],
    "Go Gin": ["gin", "gin.Default()", "package main"],
}

DATABASE_RULES = {
    "PostgreSQL": ["postgresql", "postgres", "psycopg2", "sqlalchemy"],
    "MySQL": ["mysql", "pymysql", "mysql.connector"],
    "MongoDB": ["mongodb", "pymongo", "mongoengine"],
    "Redis": ["redis", "redis-py"],
    "SQLite": ["sqlite3", "sqlite"],
}

SKELETON_BY_STACK = {
    "React": ["src/components", "src/pages", "src/routes", "src/hooks"],
    "Vue": ["src/components", "src/views", "src/router", "src/store"],
    "Angular": ["src/app", "src/components", "src/services"],
    "FastAPI": ["api/routes", "models", "schemas", "config"],
    "Django": ["apps", "models", "views", "templates", "urls.py"],
    "Flask": ["app", "routes", "templates", "static"],
    "Express": ["routes", "controllers", "models", "middleware"],
    "Spring Boot": ["src/main/java", "src/main/resources", "controllers", "services"],
}


def score_matches(text: str, rules: Dict[str, list]) -> List[Dict[str, Any]]:
    lowered = text.lower()
    items = []
    for label, keywords in rules.items():
        found = [kw for kw in keywords if kw.lower() in lowered]
        if found:
            items.append({
                "label": label,
                "count": len(found),
                "keywords": found,
            })
    return sorted(items, key=lambda x: x["count"], reverse=True)


def infer_from_text(text: str):
    frontend = score_matches(text, FRONTEND_RULES)
    backend = score_matches(text, BACKEND_RULES)
    database = score_matches(text, DATABASE_RULES)

    frontend_label = frontend[0]["label"] if frontend else None
    backend_label = backend[0]["label"] if backend else None
    database_label = database[0]["label"] if database else None

    evidence = []
    if frontend:
        evidence.extend(frontend[0]["keywords"])
    if backend:
        evidence.extend(backend[0]["keywords"])
    if database:
        evidence.extend(database[0]["keywords"])

    skeleton = []
    if frontend_label in SKELETON_BY_STACK:
        skeleton.extend(SKELETON_BY_STACK[frontend_label])
    if backend_label in SKELETON_BY_STACK:
        skeleton.extend(SKELETON_BY_STACK[backend_label])

    confidence = 0.5
    if frontend_label:
        confidence += 0.2
    if backend_label:
        confidence += 0.2
    if database_label:
        confidence += 0.1
    confidence = round(min(confidence, 0.99), 2)

    return {
        "frontend": frontend_label,
        "backend": backend_label,
        "database": database_label,
        "confidence": confidence,
        "evidence": evidence[:20],
        "skeleton": list(dict.fromkeys(skeleton))[:10],
    }


def collect_candidate_signals(url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta = " ".join([m.get("content", "") for m in soup.find_all("meta")])
    links = [a.get("href", "") for a in soup.find_all("a") if a.get("href")]
    scripts = [s.get("src", "") for s in soup.find_all("script") if s.get("src")]
    css = [l.get("href", "") for l in soup.find_all("link") if l.get("href")]
    body_text = soup.get_text(" ", strip=True)

    combined = " ".join([
        url,
        title,
        meta,
        body_text,
        " ".join(links),
        " ".join(scripts),
        " ".join(css),
    ])

    return {
        "title": title,
        "meta": meta[:300],
        "links": links[:30],
        "scripts": scripts[:30],
        "css": css[:30],
        "combined_text": combined,
    }


async def fetch_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; XILIkelong/1.0; +https://github.com/nft191881-hub/XILIkelong)"
    }
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


class TextPayload(BaseModel):
    text: str


class UrlPayload(BaseModel):
    url: str


@app.get("/")
def root():
    return {
        "message": "XILIkelong API is running",
        "status": "ok",
        "version": "0.1.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer/text")
def infer_text(payload: TextPayload):
    return infer_from_text(payload.text)


@app.post("/infer/code")
def infer_code(payload: TextPayload):
    return infer_from_text(payload.text)


@app.post("/infer/url")
async def infer_url(payload: UrlPayload):
    try:
        html = await fetch_url(payload.url)
    except Exception as exc:
        return {
            "error": f"Failed to fetch URL: {str(exc)}",
            "source_url": payload.url
        }

    signals = collect_candidate_signals(payload.url, html)
    result = infer_from_text(signals["combined_text"])

    result["source_url"] = payload.url
    result["title"] = signals.get("title")
    result["links"] = signals.get("links", [])[:10]
    result["scripts"] = signals.get("scripts", [])[:10]
    result["css"] = signals.get("css", [])[:10]

    return result
