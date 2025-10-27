from models.queries.data_agnostic import build_data_agnostic_query


def build_querent(name: str, *args, **kwargs):
    if 'data_agnostic' in name.lower():
        return build_data_agnostic_query(name, *args, **kwargs)
