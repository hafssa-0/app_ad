"""Aide à la décision : méthodes MCDM + Fuzzy TOPSIS (Streamlit)
Lancer avec :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

VIOLET = "#A855F7"

# =====================================================================
#  FONCTIONS DE CALCUL (Crisp & Fuzzy)
# =====================================================================

def classement(noms, scores, nom_col="Score", plus_grand_mieux=True):
    df = pd.DataFrame({nom_col: scores}, index=noms)
    df["Rang"] = df[nom_col].rank(ascending=not plus_grand_mieux, method="min").astype(int)
    return df


def normaliser_ratio(X, cost):
    R = X / X.max(axis=0)
    R[:, cost] = X.min(axis=0)[cost] / X[:, cost]
    return R


def normaliser_minmax(X, cost):
    mn, mx = X.min(axis=0), X.max(axis=0)
    den = np.where(mx - mn == 0, 1, mx - mn)
    R = (X - mn) / den
    R[:, cost] = ((mx - X) / den)[:, cost]
    return R


RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


def ahp_poids(A):
    n = len(A)
    s = A.sum(axis=0)
    w = (A / s).mean(axis=1)
    lam = float((s * w).sum())
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.56)
    CR = CI / ri if ri > 0 else 0.0
    return w, lam, CI, CR


def bwm_poids(n, b, w_, BO, OW):
    rows = []
    for j in range(n):
        r = np.zeros(n + 1); r[b] += 1; r[j] -= BO[j]; r[n] = -1; rows.append(r)
        r = np.zeros(n + 1); r[b] -= 1; r[j] += BO[j]; r[n] = -1; rows.append(r)
        r = np.zeros(n + 1); r[j] += 1; r[w_] -= OW[j]; r[n] = -1; rows.append(r)
        r = np.zeros(n + 1); r[j] -= 1; r[w_] += OW[j]; r[n] = -1; rows.append(r)
    c = np.zeros(n + 1); c[n] = 1
    A_eq = [np.append(np.ones(n), 0)]
    res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * (n + 1), method="highs")
    return res.x[:n], float(res.x[n])


def entropie_poids(X, cost):
    m, n = X.shape
    D = normaliser_ratio(X, cost)
    P = D / D.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -plogp.sum(axis=0) / np.log(m)
    div = 1 - E
    w = div / div.sum() if div.sum() > 0 else np.ones(n) / n
    return P, E, div, w


def critic_poids(X, cost):
    R = normaliser_minmax(X, cost)
    sigma = R.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.nan_to_num(np.corrcoef(R, rowvar=False))
    np.fill_diagonal(rho, 1.0)
    indep = (1 - np.abs(rho)).sum(axis=1)
    C = sigma * indep
    w = C / C.sum() if C.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]
    return R, rho, sigma, C, w


def wsm_wpm_waspas(X, cost, w, lam):
    R = normaliser_ratio(X, cost)
    q1 = (R * w).sum(axis=1)
    q2 = np.prod(R ** w, axis=1)
    return R, q1, q2, lam * q1 + (1 - lam) * q2


def topsis(X, cost, w):
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    ideal = np.where(cost, V.min(axis=0), V.max(axis=0))
    anti = np.where(cost, V.max(axis=0), V.min(axis=0))
    sp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    sm = np.sqrt(((V - anti) ** 2).sum(axis=1))
    rc = sm / (sp + sm)
    return R, V, ideal, anti, sp, sm, rc


def vikor(X, cost, w, v):
    best = np.where(cost, X.min(axis=0), X.max(axis=0))
    worst = np.where(cost, X.max(axis=0), X.min(axis=0))
    den = np.where(best - worst == 0, 1, best - worst)
    T = w * (best - X) / den
    S, R = T.sum(axis=1), T.max(axis=1)
    def part(a):
        d = a.max() - a.min()
        return (a - a.min()) / d if d > 0 else np.zeros_like(a)
    Q = v * part(S) + (1 - v) * part(R)
    return S, R, Q


def spearman(r1, r2):
    n = len(r1)
    d = np.asarray(r1) - np.asarray(r2)
    return 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))


# --- Fuzzy TOPSIS Fonctions ---
def distance_tfn(t1, t2):
    return np.sqrt(1/3 * ((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2 + (t1[2] - t2[2])**2))


def fuzzy_topsis_calc(X_fuzzy, w_fuzzy, cost_mask):
    """
    X_fuzzy: ndarray de forme (m, n, 3)
    w_fuzzy: ndarray de forme (n, 3)
    cost_mask: array de booléens (True si coût)
    """
    m, n, _ = X_fuzzy.shape
    R = np.zeros_like(X_fuzzy, dtype=float)

    for j in range(n):
        if cost_mask[j]:
            a_min = X_fuzzy[:, j, 0].min()
            for i in range(m):
                a, b, c = X_fuzzy[i, j]
                R[i, j] = (a_min / max(c, 1e-6), a_min / max(b, 1e-6), a_min / max(a, 1e-6))
        else:
            c_max = X_fuzzy[:, j, 2].max()
            for i in range(m):
                a, b, c = X_fuzzy[i, j]
                R[i, j] = (a / max(c_max, 1e-6), b / max(c_max, 1e-6), c / max(c_max, 1e-6))

    V = np.zeros_like(R)
    for i in range(m):
        for j in range(n):
            wa, wb, wc = w_fuzzy[j]
            ra, rb, rc = R[i, j]
            V[i, j] = (ra * wa, rb * wb, rc * wc)

    v_star = np.array([1.0, 1.0, 1.0])
    v_minus = np.array([0.0, 0.0, 0.0])

    D_star = np.zeros(m)
    D_minus = np.zeros(m)
    for i in range(m):
        d_s = sum(distance_tfn(V[i, j], v_star) for j in range(n))
        d_m = sum(distance_tfn(V[i, j], v_minus) for j in range(n))
        D_star[i] = d_s
        D_minus[i] = d_m

    CC = D_minus / (D_minus + D_star + 1e-10)
    return R, V, D_star, D_minus, CC


# =====================================================================
#  OUTILS D'INTERFACE
# =====================================================================

def donnees():
    d = st.session_state.get("data")
    if d is None:
        st.warning("Commence par enregistrer une matrice dans « Accueil / Matrice de décision ».")
        return None
    X = d["df"].to_numpy(dtype=float)
    cost = np.array([c in d["couts"] for c in d["crits"]])
    with st.expander("Voir la matrice utilisée"):
        st.dataframe(d["df"])
        st.caption("Critères à minimiser : " + (", ".join(d["couts"]) if d["couts"] else "aucun"))
    return d["alts"], d["crits"], X, cost


def exiger_positif(X):
    if (X <= 0).any():
        st.error("Cette méthode demande des valeurs strictement positives dans la matrice.")
        return False
    return True


def bloc_poids(crits, key):
    n = len(crits)
    stock = st.session_state.setdefault("poids", {})
    choix = st.selectbox("Poids des critères", ["Poids égaux", "Saisie manuelle"] + list(stock), key=key + "_src")
    if choix == "Poids égaux":
        w = np.ones(n)
    elif choix == "Saisie manuelle":
        cols = st.columns(n)
        w = np.array([cols[i].number_input(crits[i], 0.0, 1000.0, 1.0, 0.1, key=f"{key}_w{i}") for i in range(n)])
    else:
        w = np.array(stock[choix], dtype=float)
    if w.sum() == 0:
        w = np.ones(n)
    w = w / w.sum()
    st.caption("Poids utilisés : " + " | ".join(f"{c} = {x:.3f}" for c, x in zip(crits, w)))
    return w


def graphique_poids(crits, w):
    st.bar_chart(pd.Series(w, index=crits, name="Poids"), color=VIOLET)


def sauver_poids(nom, crits, w):
    st.session_state.setdefault("poids", {})[nom] = w


def aff(df, d=4):
    st.dataframe(df.round(d))


def meilleure(df, col):
    st.success(f"Meilleure alternative : **{df.index[df['Rang'] == 1][0]}**")


ECHELLE = [1 / k for k in range(9, 1, -1)] + [float(k) for k in range(1, 10)]


def fmt(v):
    return f"1/{round(1 / v)}" if v < 1 else str(int(v))


def comparaisons(noms, key, phrase):
    n = len(noms)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = st.select_slider(phrase.format(a=noms[i], b=noms[j]), options=ECHELLE, value=1.0,
                                 format_func=fmt, key=f"{key}_{i}_{j}")
            A[i, j], A[j, i] = v, 1 / v
    return A


# =====================================================================
#  PAGES
# =====================================================================

def page_accueil():
    st.title("Aide à la décision : méthodes MCDM")
    st.caption("App réalisée dans le cadre du cours Aide à la décision")

    with st.expander("Rappel : qu'est-ce que le MCDM ?"):
        st.markdown("""
Un décideur doit choisir parmi plusieurs **alternatives** selon plusieurs **critères** souvent contradictoires.
Cette application intègre à la fois les méthodes classiques (**Crisp MCDM**) et les méthodes sous incertitude (**Fuzzy TOPSIS**).
""")

    st.header("Matrice de décision")
    st.write("Renseignez les alternatives, les critères et les valeurs correspondantes.")

    alts_txt = st.text_area("Alternatives (une par ligne)", "A1\nA2\nA3", height=100)
    crits_txt = st.text_area("Critères (un par ligne)", "C1\nC2\nC3", height=100)
    alts = [l.strip() for l in alts_txt.splitlines() if l.strip()]
    crits = [l.strip() for l in crits_txt.splitlines() if l.strip()]

    if len(alts) < 2 or len(crits) < 2 or len(set(alts)) < len(alts) or len(set(crits)) < len(crits):
        st.error("Il faut au moins 2 alternatives et 2 critères, avec des noms différents.")
        return

    saved = st.session_state.get("data")
    if saved and saved["alts"] == alts and saved["crits"] == crits:
        base = saved["df"].copy()
    else:
        base = pd.DataFrame(1.0, index=alts, columns=crits)

    st.write("**Valeurs de la matrice (Crisp) :**")
    df = st.data_editor(base, key="ed_" + "|".join(alts) + "#" + "|".join(crits))
    couts = st.multiselect("Critères à minimiser (coût)", crits,
                           default=[c for c in (saved["couts"] if saved else []) if c in crits])

    if st.button("Enregistrer la matrice"):
        if df.isna().any().any():
            st.error("Toutes les cases doivent être remplies.")
        else:
            st.session_state["data"] = {"alts": alts, "crits": crits, "df": df.astype(float), "couts": couts}
            st.session_state["poids"] = {}
            st.success("Matrice enregistrée. Choisis une méthode dans le menu de gauche.")


# ---------------------------------------------------------------- AHP
def page_ahp():
    st.title("AHP : Analytic Hierarchy Process")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode **subjective** : l'expert compare les critères **deux à deux** avec l'échelle de Saaty.
""")
        st.latex(r"\lambda_{max}=\sum_i s_i\,w_i \qquad CI=\frac{\lambda_{max}-n}{n-1} \qquad CR=\frac{CI}{RI}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d

    st.subheader("1) Comparaison des critères")
    A = comparaisons(crits, "ahp_c", "**{a}** est ... fois plus important que **{b}**")
    w, lam, CI, CR = ahp_poids(A)
    aff(pd.DataFrame(A, index=crits, columns=crits), 3)
    res = pd.DataFrame({"Poids": w}, index=crits)
    aff(res)
    graphique_poids(crits, w)
    st.write(f"λmax = **{lam:.4f}**   |   CI = **{CI:.4f}**   |   CR = **{CR:.4f}**")
    if CR < 0.1:
        st.success("CR < 0,1 : les jugements sont cohérents.")
    else:
        st.error("CR ≥ 0,1 : jugements incohérents.")
    sauver_poids("AHP", crits, w)


# ---------------------------------------------------------------- BWM
def page_bwm():
    st.title("BWM : Best-Worst Method")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode **subjective** (Rezaei, 2015). Moins de comparaisons requises ($2n-3$).
""")
        st.latex(r"\min \xi \quad \text{s.c.}\quad |w_B - a_{Bj}\,w_j|\le\xi,\quad |w_j - a_{jW}\,w_W|\le\xi,\quad \sum_j w_j=1")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    n = len(crits)
    c1, c2 = st.columns(2)
    best = c1.selectbox("Critère le plus important (Best)", crits, index=0)
    worst = c2.selectbox("Critère le moins important (Worst)", crits, index=n - 1)
    if best == worst:
        st.error("Best et Worst doivent être différents.")
        return
    b, wi = crits.index(best), crits.index(worst)

    BO = np.ones(n)
    cols = st.columns(n)
    for j in range(n):
        if j != b:
            BO[j] = cols[j].number_input(f"{best} / {crits[j]}", 1, 9, 2, key=f"bo{j}")
    OW = np.ones(n)
    cols = st.columns(n)
    for j in range(n):
        if j != wi:
            OW[j] = cols[j].number_input(f"{crits[j]} / {worst}", 1, 9, 2, key=f"ow{j}")

    w, xi = bwm_poids(n, b, wi, BO, OW)
    aff(pd.DataFrame({"Poids": w}, index=crits))
    graphique_poids(crits, w)
    st.write(f"ξ* = **{xi:.4f}**")
    sauver_poids("BWM", crits, w)


# ---------------------------------------------------------------- Entropie
def page_entropie():
    st.title("Entropie")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Mesure du désordre et de la dispersion objective des données.")
        st.latex(r"E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij} \qquad w_j=\frac{1-E_j}{n-\sum_k E_k}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    if not exiger_positif(X):
        return
    P, E, div, w = entropie_poids(X, cost)
    aff(pd.DataFrame({"Entropie Ej": E, "1 - Ej": div, "Poids wj": w}, index=crits))
    graphique_poids(crits, w)
    sauver_poids("Entropie", crits, w)


# ---------------------------------------------------------------- CRITIC
def page_critic():
    st.title("CRITIC")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Prend en compte l'écart-type (dispersion) et l'indépendance (corrélations).")
        st.latex(r"C_j=\sigma_j\sum_{k=1}^{n}\left(1-|\rho_{jk}|\right) \qquad w_j=\frac{C_j}{\sum_j C_j}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    R, rho, sigma, C, w = critic_poids(X, cost)
    aff(pd.DataFrame({"Écart-type σj": sigma, "Indice Cj": C, "Poids wj": w}, index=crits))
    graphique_poids(crits, w)
    sauver_poids("CRITIC", crits, w)


# ---------------------------------------------------------------- WSM / WPM / WASPAS
def page_waspas():
    st.title("WSM - WPM - WASPAS")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Méthodes additives et multiplicatives combinées.")
        st.latex(r"Q_i=\lambda\,\sum w_j r_{ij} + (1-\lambda)\,\prod (r_{ij})^{w_j}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    if not exiger_positif(X):
        return
    w = bloc_poids(crits, "wsm")
    lam = st.slider("λ (WASPAS)", 0.0, 1.0, 0.5, 0.05)
    R, q1, q2, q = wsm_wpm_waspas(X, cost, w, lam)
    out = pd.DataFrame({"WSM": q1, "WPM": q2, "WASPAS": q}, index=alts)
    for m in ["WSM", "WPM", "WASPAS"]:
        out["Rang " + m] = out[m].rank(ascending=False, method="min").astype(int)
    aff(out)


# ---------------------------------------------------------------- TOPSIS
def page_topsis():
    st.title("TOPSIS")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Distance à la solution idéale positive et négative.")
        st.latex(r"RC_i^*=\frac{S_i^-}{S_i^+ + S_i^-}\in[0,1]")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    w = bloc_poids(crits, "top")
    R, V, ideal, anti, sp, sm, rc = topsis(X, cost, w)
    out = pd.DataFrame({"S+": sp, "S-": sm, "RC*": rc}, index=alts)
    out["Rang"] = out["RC*"].rank(ascending=False, method="min").astype(int)
    aff(out)
    meilleure(out, "RC*")


# ---------------------------------------------------------------- VIKOR
def page_vikor():
    st.title("VIKOR")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Classement de compromis basé sur l'utilité de groupe (S) et le regret individuel (R).")
        st.latex(r"Q_i=v\,\frac{S_i-S^*}{S^--S^*}+(1-v)\,\frac{R_i-R^*}{R^--R^R}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    w = bloc_poids(crits, "vik")
    v = st.slider("v (stratégie de groupe)", 0.0, 1.0, 0.5, 0.05)
    S, R, Q = vikor(X, cost, w, v)
    out = pd.DataFrame({"S": S, "R": R, "Q": Q}, index=alts)
    for c in "SRQ":
        out["Rang " + c] = out[c].rank(method="min").astype(int)
    aff(out)


# ---------------------------------------------------------------- FUZZY TOPSIS
def page_fuzzy_topsis():
    st.title("Fuzzy TOPSIS (Nombres Flous Triangulaires)")
    with st.expander("Principe et cours (Chen, 2000)", expanded=True):
        st.markdown("""
**Principe.** Quand l'information est imprécise ou subjective, on utilise des **nombres flous triangulaires (TFN)** notés $\\tilde{x} = (a, b, c)$[cite: 136, 137] :
- $a$ : valeur pessimiste
- $b$ : valeur la plus probable
- $c$ : valeur optimiste

Les étapes reproduisent le TOPSIS classique avec l'arithmétique floue (normalisation, pondération par des poids flous, distances euclidiennes floues et coefficient de proximité $CC_i$).
""")

    st.subheader("Configuration des données floues")
    alts_input = st.text_area("Alternatives (une par ligne)", "Alternatif_1\nAlternatif_2\nAlternatif_3", height=70)
    crits_input = st.text_area("Critères (un par ligne)", "C1\nC2\nC3", height=70)

    alts = [a.strip() for a in alts_input.splitlines() if a.strip()]
    crits = [c.strip() for c in crits_input.splitlines() if c.strip()]

    if len(alts) < 2 or len(crits) < 2:
        st.warning("Veuillez saisir au moins 2 alternatives et 2 critères.")
        return

    couts_fuzzy = st.multiselect("Critères de type Coût (à minimiser)", crits, key="ft_costs")

    st.markdown("### Saisie de la matrice de décision floue (Format: `a, b, c`)")
    st.caption("Exemple : `1, 3, 5` pour un nombre flou triangulaire.")

    fuzzy_matrix = np.zeros((len(alts), len(crits), 3))
    cols = st.columns(len(crits))

    # Tableau de saisie simplifié via texte ou valeurs par défaut
    for j, c in enumerate(crits):
        cols[j].markdown(f"**{c}**")
        for i, a in enumerate(alts):
            default_val = f"{i+1+j}, {i+2+j}, {i+3+j}"
            val_str = cols[j].text_input(f"{a} ({c})", value=default_val, key=f"ft_{i}_{j}")
            try:
                parts = [float(p.strip()) for p in val_str.split(",")]
                if len(parts) == 3:
                    fuzzy_matrix[i, j] = parts
                else:
                    fuzzy_matrix[i, j] = [1, 2, 3]
            except:
                fuzzy_matrix[i, j] = [1, 2, 3]

    st.markdown("### Poids des critères (Nombres flous triangulaires)")
    weight_matrix = np.zeros((len(crits), 3))
    w_cols = st.columns(len(crits))
    for j, c in enumerate(crits):
        w_str = w_cols[j].text_input(f"Poids {c}", value="0.2, 0.3, 0.5", key=f"ft_w_{j}")
        try:
            parts = [float(p.strip()) for p in w_str.split(",")]
            weight_matrix[j] = parts if len(parts) == 3 else [0.3, 0.3, 0.4]
        except:
            weight_matrix[j] = [0.3, 0.3, 0.4]

    if st.button("Lancer le calcul Fuzzy TOPSIS"):
        cost_mask = np.array([c in couts_fuzzy for c in crits])
        R, V, D_star, D_minus, CC = fuzzy_topsis_calc(fuzzy_matrix, weight_matrix, cost_mask)

        res_df = pd.DataFrame({
            "D* (Distance Idéal Positif)": D_star,
            "D- (Distance Idéal Négatif)": D_minus,
            "Coefficient CCi": CC
        }, index=alts)
        res_df["Rang"] = res_df["Coefficient CCi"].rank(ascending=False, method="min").astype(int)

        st.success("Calculs terminés avec succès !")
        aff(res_df.sort_values("Rang"))
        meilleure(res_df, "Coefficient CCi")


# ---------------------------------------------------------------- Comparaison
def page_comparaison():
    st.title("Comparaison des classements (Spearman)")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Mesure l'accord entre les différents classements obtenus par les méthodes classiques.")
        st.latex(r"\rho=1-\frac{6\sum d_i^2}{n(n^2-1)}")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    if not exiger_positif(X):
        return
    w = bloc_poids(crits, "cmp")
    _, q1, q2, q = wsm_wpm_waspas(X, cost, w, 0.5)
    rc = topsis(X, cost, w)[-1]
    Q = vikor(X, cost, w, 0.5)[2]
    scores = {"WSM": q1, "WPM": q2, "WASPAS": q, "TOPSIS": rc, "VIKOR": -Q}
    rangs = pd.DataFrame({k: pd.Series(s, index=alts).rank(ascending=False, method="min") for k, s in scores.items()}).astype(int)
    st.dataframe(rangs)


# ---------------------------------------------------------------- MAUT
def page_maut():
    st.title("Alternative unique (MAUT)")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("Évaluation absolue d'une seule alternative par rapport à des seuils d'utilité.")
        st.latex(r"S(A)=\sum_j w_j\,u_j(x_j)")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    w = bloc_poids(crits, "maut")
    u = np.zeros(len(crits))
    for j, c in enumerate(crits):
        a, b_, e = st.columns(3)
        x = a.number_input(f"{c} : valeur", value=float(X[:, j].mean()), key=f"mx{j}")
        mn = b_.number_input(f"{c} : min", value=float(X[:, j].min()), key=f"mn{j}")
        mx = e.number_input(f"{c} : max", value=float(X[:, j].max()), key=f"mm{j}")
        if mx > mn:
            u[j] = (mx - x) / (mx - mn) if cost[j] else (x - mn) / (mx - mn)
        u[j] = min(max(u[j], 0.0), 1.0)
    tau = st.slider("Seuil de décision τ", 0.0, 1.0, 0.7, 0.01)
    S = float((w * u).sum())
    aff(pd.DataFrame({"Utilité uj": u, "Poids wj": w}, index=crits))
    st.write(f"Score **S(A) = {S:.4f}**")
    if S >= tau:
        st.success("Décision : ACCEPTER")
    else:
        st.error("Décision : REJETER")


# =====================================================================
#  APPLICATION NAVIGATION
# =====================================================================

PAGES = {
    "Accueil / Matrice de décision": page_accueil,
    "AHP": page_ahp,
    "BWM": page_bwm,
    "Entropie": page_entropie,
    "CRITIC": page_critic,
    "WSM - WPM - WASPAS": page_waspas,
    "TOPSIS": page_topsis,
    "VIKOR": page_vikor,
    "Fuzzy TOPSIS": page_fuzzy_topsis,
    "Comparaison (Spearman)": page_comparaison,
    "Alternative unique (MAUT)": page_maut,
}


def main():
    st.set_page_config(page_title="Aide à la décision MCDM", page_icon="🟣", layout="wide")
    st.markdown(f"<style>h1,h2,h3{{color:{VIOLET};}}</style>", unsafe_allow_html=True)
    choix = st.sidebar.radio("Choisir une méthode", list(PAGES))
    PAGES[choix]()


if __name__ == "__main__":
    main()