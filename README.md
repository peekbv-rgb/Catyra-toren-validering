# Catyra — Kasklimaat Simulator

Interactief webdashboard voor het simuleren van kasklimaat met verdampings­koeltorens.
Gebouwd op een gekalibreerde fysica-engine en met alle productie-analyses ingebouwd.

Ontwikkeld door **Catyra Engineering** · versie 2.0 (2026)

---

## Wat zit erin?

| Bestand | Omschrijving |
| --- | --- |
| `app.py` | Streamlit-dashboard (klimaatanalyse, weerbericht, historische VD, toren-simulatie) |
| `greenhouse_climate_v2.py` | Fysica-engine (kas + boomgaard + toren) |
| `demo_data.xlsx` | Voorbeelddata om mee te testen |
| `requirements.txt` | Python-dependencies |

## Functionaliteit

- **Twee hoofdtabs:** Jaaranalyse en Weerbericht (7-daags met torensimulatie).
- **Tab Toren-performance:** losse berekening van de prestatie per toren — met name de **luchthoeveelheid** (m³/hr) uit het schoorsteeneffect, plus uitlaatsnelheid, ΔT, verdamping, koelvermogen en luchtwisselingen; met grafiek en tabel over een temperatuurbereik.
- **Analyses in de Jaaranalyse:** Hot Day, Annual, Damage, Production, Temperature, DIF, Economic en Export.
- **Torenbesturing** op basis van VD (vochtdeficiet) met hysterese en dakraam-regeling met buiten-RV-bescherming.
- **Historische VD-percentielen** uit meerjarige data.
- Alle toreninstellingen (hoogte, C-waarde, diameter, VD-drempels) instelbaar in de sidebar.
- Data-upload voor klimaatdata (xlsx/csv), productiedata, 7-daags weerbericht en meerjarige historische data.

## Installatie

Vereist Python 3.10 of hoger.

```bash
# 1. (aanbevolen) virtuele omgeving
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. dependencies installeren
pip install -r requirements.txt
```

## Starten

```bash
streamlit run app.py --server.port 8510
```

De browser opent automatisch op <http://localhost:8510>. Upload `demo_data.xlsx` via de sidebar om direct een simulatie te draaien.

## Publiceren op Render

Dit is een Streamlit-app, dus **geen statische site maar een Web Service** (een draaiende Python-server). Er is geen HTML-bestand nodig — Streamlit genereert de interface zelf.

De meegeleverde `render.yaml` (Blueprint) regelt alles. Via het Render-dashboard:

1. **New → Blueprint**, koppel de GitHub-repo. Render leest `render.yaml` en zet de service op.
2. Of handmatig via **New → Web Service**:
   - Language: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Belangrijk:
- Streamlit moet aan `$PORT` van Render binden (staat al goed in het startcommando en `render.yaml`).
- Op het gratis plan **valt de service in slaap na ~15 min inactiviteit** en wordt bij het volgende bezoek weer wakker — de eerste laadbeurt duurt dan wat langer.
- `.streamlit/config.toml` staat op `headless = true` en `enableCORS = false` zodat het achter de proxy van Render werkt.

## Gebruik

1. **Sidebar** — upload klimaatdata (xlsx/csv), optioneel productiedata, een 7-daags weerbericht (CSV) en meerjarige historische data (CSV).
2. **Tab Jaaranalyse** — volledige klimaatsimulatie met toren plus historische VD-percentielen.
3. **Tab Weerbericht** — 7-daagse vooruitblik met torensimulatie.
4. Stel de torenparameters (hoogte, C-waarde, diameter, VD-drempels) in de sidebar bij en de analyses werken realtime mee.

## Projectstructuur

```
.
├── app.py                     # Streamlit-dashboard
├── greenhouse_climate_v2.py   # Fysica-engine
├── demo_data.xlsx             # Voorbeelddata
├── requirements.txt           # Dependencies
├── render.yaml                # Render deploy-config (Blueprint)
├── .python-version            # Python-versie voor Render
├── .streamlit/
│   └── config.toml            # Streamlit server-instellingen
├── .gitignore
└── README.md
```

## Licentie

Nog niet bepaald. Voeg een `LICENSE`-bestand toe om de gebruiksvoorwaarden vast te leggen
(bijv. MIT voor open source, of een propriëtaire tekst voor intern gebruik).

---

© 2026 Catyra Engineering
