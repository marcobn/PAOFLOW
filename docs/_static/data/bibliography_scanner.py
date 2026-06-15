import json
from pathlib import Path

import requests

OUTPUT_FILE = Path(__file__).parent / 'paoflow-used-by.json'

PAOFLOW_DOIS = [
    '10.1016/j.commatsci.2021.110828',
    '10.1016/j.commatsci.2017.11.034',
]

works = {}

for doi in PAOFLOW_DOIS:
    openalex_url = f'https://api.openalex.org/works/https://doi.org/{doi}'
    cited_work = requests.get(openalex_url).json()['id']

    cursor = '*'
    while cursor:
        r = requests.get(
            'https://api.openalex.org/works',
            params={
                'filter': f'cites:{cited_work}',
                'per-page': 200,
                'cursor': cursor,
            },
        ).json()

        for work in r['results']:
            title = work.get('title') or ''
            abstract_index = work.get('abstract_inverted_index') or {}
            abstract_words = ' '.join(abstract_index.keys())

            searchable_text = f'{title} {abstract_words}'.lower()

            if 'paoflow' in searchable_text or 'pao-flow' in searchable_text:
                doi_value = work.get('doi')
                if doi_value:
                    clean_doi = doi_value.replace('https://doi.org/', '')
                    works[clean_doi] = work

        cursor = r['meta'].get('next_cursor')
        if not r['results']:
            break

records = []

for doi, work in works.items():
    records.append(
        {
            'year': work.get('publication_year'),
            'title': work.get('title'),
            'journal': ((work.get('primary_location') or {}).get('source') or {}).get(
                'display_name'
            ),
            'doi': doi,
            'url': f'https://doi.org/{doi}',
            'authors': [
                author['author']['display_name']
                for author in work.get('authorships', [])
                if author.get('author')
            ],
        }
    )

records.sort(key=lambda item: item.get('year') or 0, reverse=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
