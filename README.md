# URLSecure / BankSecure – Streamlit app

A csomag tartalmazza a futó alkalmazáshoz szükséges fájlokat:

- `app.py`
- `app_stage1_pro.py`
- `stage1_model.joblib`
- `stage1_meta.json`
- `white_list_tranco_clean_for_a_zone.csv`
- `requirements.txt`
- `.streamlit/config.toml`

## Streamlit Community Cloud
A deployhoz az entrypoint fájl: `app.py`

Ha Web Risk API-t is használsz, a kulcsot Streamlit secretsként add meg:
`GOOGLE_CLOUD_API_KEY`
