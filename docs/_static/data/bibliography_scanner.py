import json
import time
from pathlib import Path

import requests

OUTPUT_FILE = Path(__file__).parent / 'paoflow-used-by.json'

PAOFLOW_DOIS = [
    '10.1016/j.commatsci.2021.110828',
    '10.1016/j.commatsci.2017.11.034',
]

SEARCH_TERMS = ('paoflow', 'pao-flow', 'pao flow')

session = requests.Session()
session.headers.update({'User-Agent': 'PAOFLOW bibliography updater; mailto:YOUR_EMAIL_HERE'})


def abstract_from_inverted_index(index):
    if not index:
        return ''

    words = []
    for word, positions in index.items():
        for position in positions:
            words.append((position, word))

    return ' '.join(word for _, word in sorted(words))


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_doi(doi):
    if not doi:
        return None

    return doi.lower().replace('https://doi.org/', '').strip()


def collect_citing_works(doi):
    cited_work = get_json(f'https://api.openalex.org/works/https://doi.org/{doi}')['id']

    cursor = '*'
    citing_works = {}

    while cursor:
        data = get_json(
            'https://api.openalex.org/works',
            params={
                'filter': f'cites:{cited_work}',
                'per-page': 200,
                'cursor': cursor,
            },
        )

        for work in data['results']:
            work_doi = normalize_doi(work.get('doi'))
            key = work_doi or work['id']
            citing_works[key] = work

        cursor = data['meta'].get('next_cursor')

        if not data['results']:
            break

        time.sleep(0.1)

    return citing_works


works = {}

for doi in PAOFLOW_DOIS:
    works.update(collect_citing_works(doi))


records = []

for key, work in works.items():
    title = work.get('title') or ''
    abstract = abstract_from_inverted_index(work.get('abstract_inverted_index'))
    searchable_text = f'{title} {abstract}'.lower()

    doi = normalize_doi(work.get('doi'))
    used_paoflow_in_metadata = any(term in searchable_text for term in SEARCH_TERMS)

    records.append(
        {
            'year': work.get('publication_year'),
            'title': title,
            'journal': ((work.get('primary_location') or {}).get('source') or {}).get(
                'display_name'
            ),
            'doi': doi,
            'url': f'https://doi.org/{doi}' if doi else work.get('id'),
            'authors': [
                authorship['author']['display_name']
                for authorship in work.get('authorships', [])
                if authorship.get('author')
            ],
            'openalex_id': work.get('id'),
            'cited_by_count': work.get('cited_by_count'),
            'used_paoflow_in_metadata': used_paoflow_in_metadata,
            'include_on_website': used_paoflow_in_metadata,
        }
    )

records.sort(
    key=lambda item: (
        item.get('year') or 0,
        item.get('title') or '',
    ),
    reverse=True,
)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f'Wrote {len(records)} citing works to {OUTPUT_FILE}')
print(
    f'{sum(record["used_paoflow_in_metadata"] for record in records)} explicitly mention PAOFLOW in title/abstract metadata'
)
