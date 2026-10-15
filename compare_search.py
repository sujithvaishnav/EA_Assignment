
import os
import time
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from exa_py import Exa

load_dotenv()

QUERY = (
    "LangGraph conditional edges query rewriting "
    "self-corrective RAG official documentation"
)


def search_tavily(query):
    client = TavilyClient(
        api_key=os.environ["TAVILY_API_KEY"]
    )
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_raw_content=False,
    )

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score"),
        }
        for r in response.get("results", [])
    ]


def search_serper(query):
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": query, "num": 5},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "content": r.get("snippet", ""),
            "score": None,
        }
        for r in data.get("organic", [])
    ]


def search_exa(query):
    client = Exa(api_key=os.environ["EXA_API_KEY"])
    response = client.search(
        query,
        type="auto",
        num_results=5,
        contents={"highlights": True},
    )

    results = []
    for r in response.results:
        highlights = getattr(r, "highlights", None) or []
        results.append({
            "title": r.title or "",
            "url": r.url or "",
            "content": " ".join(highlights),
            "score": getattr(r, "score", None),
        })

    return results


providers = {
    "Tavily": search_tavily,
    "Serper": search_serper,
    "Exa": search_exa,
}

for name, search_fn in providers.items():
    print(f"\n{'=' * 70}")
    print(name)
    print("=" * 70)

    start = time.perf_counter()

    try:
        results = search_fn(QUERY)
        elapsed = time.perf_counter() - start

        print(f"Results: {len(results)}")
        print(f"Client-observed time: {elapsed:.2f}s")

        for rank, item in enumerate(results, 1):
            print(f"\n{rank}. {item['title']}")
            print(f"URL: {item['url']}")
            print(f"Content: {item['content'][:400]}")
            print(f"Score: {item['score']}")

    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
