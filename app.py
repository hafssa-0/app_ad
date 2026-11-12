"""Aide à la décision : méthodes MCDM (Streamlit)
Lancer avec :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

VIOLET = "#A855F7"

# =====================================================================
#  FONCTIONS DE CALCUL (indépendantes de l'interface)
# =====================================================================

def classement(noms, scores, nom_col="Score", plus_grand_mieux=True):
    df = pd.DataFrame({nom_col: scores}, index=noms)
    df["Rang"] = df[nom_col].rank(ascending=not plus_grand_mieux, method="min").astype(int)
    return df


def normaliser_ratio(X, cost):
    """x/max (benefit) ou min/x (cost) -- méthodes WSM/WPM/WASPAS et entropie."""
    R = X / X.max(axis=0)
    R[:, cost] = X.min(axis=0)[cost] / X[:, cost]
    return R


def normaliser_minmax(X, cost):
    """Normalisation min-max utilisée par CRITIC."""
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
    """Programme linéaire : min xi  avec  |wB - wj*aBj| <= xi  et  |wj - wW*ajW| <= xi."""
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


# =====================================================================
#  OUTILS D'INTERFACE
# =====================================================================

def donnees():
    """Retourne la matrice enregistrée ou None (avec un message)."""
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
    """Choix des poids : égaux, manuels, ou issus d'une autre méthode déjà calculée."""
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
    """Matrice de comparaison par paires (échelle de Saaty)."""
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
Un décideur doit choisir parmi plusieurs **alternatives** selon plusieurs **critères** souvent contradictoires
(coût, qualité, risque…). Aucune alternative n'est la meilleure partout : il faut **pondérer** les critères et
trouver un **compromis**. C'est le rôle des méthodes MCDM (Multi-Criteria Decision Making).

La matrice de décision contient $x_{ij}$ = score de l'alternative $A_i$ pour le critère $C_j$.

**Ordre d'utilisation :**
1. **Pondérer** les critères : subjectif (AHP, BWM) ou objectif (Entropie, CRITIC).
2. **Classer** les alternatives : WSM, WPM, WASPAS, TOPSIS, VIKOR.
3. **Comparer** les classements (corrélation de Spearman) et faire une analyse de sensibilité.

**Critères *benefit*** : plus c'est grand, mieux c'est. **Critères *cost*** : plus c'est petit, mieux c'est.
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

    st.write("**Valeurs de la matrice :**")
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
**Principe.** Méthode **subjective** : l'expert compare les critères **deux à deux** avec l'échelle de Saaty
(1 = égale importance, 9 = extrêmement plus important). On obtient une matrice $A$ avec $a_{ji} = 1/a_{ij}$.

**Poids (méthode approximative).**
1. Somme de chaque colonne : $s_j = \\sum_i a_{ij}$
2. Diviser chaque élément par la somme de sa colonne
3. Le poids $w_i$ est la **moyenne de chaque ligne**
""")
        st.latex(r"\lambda_{max}=\sum_i s_i\,w_i \qquad CI=\frac{\lambda_{max}-n}{n-1} \qquad CR=\frac{CI}{RI}")
        st.markdown("""
**Test de consistance.** Si $CR < 0{,}1$ les jugements sont cohérents et les poids sont conservés.
($RI$ : 0,58 pour n=3 ; 0,90 pour n=4 ; 1,12 pour n=5 ; …)

**Classement des alternatives.** On compare les alternatives deux à deux **pour chaque critère** (priorités locales),
puis on combine avec les poids :
""")
        st.latex(r"P_i=\sum_j w_j\,w_{ij}")
        st.markdown("""
**Pourquoi l'utiliser ?** Quand les poids viennent du **jugement d'un expert** et qu'on veut **vérifier sa cohérence**.
*Limite* : beaucoup de comparaisons, de l'ordre de $n(n-1)/2$.
""")
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
        st.error("CR ≥ 0,1 : jugements incohérents, révise tes comparaisons.")
    sauver_poids("AHP", crits, w)

    st.subheader("2) Classement des alternatives (optionnel)")
    if st.checkbox("Comparer aussi les alternatives critère par critère"):
        local = np.zeros((len(alts), len(crits)))
        for k, c in enumerate(crits):
            with st.expander(f"Alternatives selon le critère {c}"):
                Ak = comparaisons(alts, f"ahp_a{k}", "Selon **" + c + "** : **{a}** est ... fois meilleure que **{b}**")
                wk, _, _, crk = ahp_poids(Ak)
                local[:, k] = wk
                st.write(f"CR = {crk:.4f} " + ("(cohérent)" if crk < 0.1 else "(incohérent)"))
        final = local @ w
        out = pd.DataFrame(local, index=alts, columns=crits)
        out["Priorité finale"] = final
        out["Rang"] = out["Priorité finale"].rank(ascending=False, method="min").astype(int)
        aff(out)
        meilleure(out, "Priorité finale")


# ---------------------------------------------------------------- BWM
def page_bwm():
    st.title("BWM : Best-Worst Method")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode **subjective** (Rezaei, 2015). L'expert choisit le critère **le meilleur (Best)** et le
**pire (Worst)**, puis ne compare chaque critère qu'à ces deux repères (échelle 1 à 9) :
- vecteur **Best-to-Others** : $a_{Bj}$
- vecteur **Others-to-Worst** : $a_{jW}$

Idéalement $w_B/w_j = a_{Bj}$ et $w_j/w_W = a_{jW}$. En pratique on cherche les poids qui minimisent
l'écart maximal $\\xi$ : c'est un **programme linéaire**.
""")
        st.latex(r"\min \xi \quad \text{s.c.}\quad |w_B - a_{Bj}\,w_j|\le\xi,\quad |w_j - a_{jW}\,w_W|\le\xi,\quad \sum_j w_j=1,\ w_j\ge 0")
        st.markdown("""
$\\xi^*$ proche de 0 = jugements cohérents.

**Pourquoi l'utiliser ?** Il demande **moins de comparaisons** que l'AHP : $2n-3$ contre $n(n-1)/2$,
et donne des jugements plus cohérents.
""")
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

    st.write("**Best-to-Others** : combien de fois le Best est plus important que chaque critère (1 à 9)")
    BO = np.ones(n)
    cols = st.columns(n)
    for j in range(n):
        if j != b:
            BO[j] = cols[j].number_input(f"{best} / {crits[j]}", 1, 9, 2, key=f"bo{j}")
    st.write("**Others-to-Worst** : combien de fois chaque critère est plus important que le Worst (1 à 9)")
    OW = np.ones(n)
    cols = st.columns(n)
    for j in range(n):
        if j != wi:
            OW[j] = cols[j].number_input(f"{crits[j]} / {worst}", 1, 9, 2, key=f"ow{j}")

    if BO[wi] < BO.max():
        st.warning("Contrôle : la note la plus élevée du vecteur BO devrait porter sur le critère Worst.")
    w, xi = bwm_poids(n, b, wi, BO, OW)
    aff(pd.DataFrame({"Poids": w}, index=crits))
    graphique_poids(crits, w)
    st.write(f"ξ* = **{xi:.4f}** " + ("(proche de 0 : jugements cohérents)" if xi < 0.1 else "(assez élevé : jugements peu cohérents)"))
    sauver_poids("BWM", crits, w)


# ---------------------------------------------------------------- Entropie
def page_entropie():
    st.title("Entropie")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode **objective** : les poids sont calculés **à partir des données**. L'entropie mesure le
désordre d'une colonne :
- entropie élevée (proche de 1) : valeurs presque égales, le critère distingue mal les alternatives, **poids faible** ;
- entropie faible : valeurs dispersées, le critère discrimine bien, **poids élevé**.
""")
        st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}} \qquad E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij} \qquad w_j=\frac{1-E_j}{n-\sum_k E_k}")
        st.markdown("""
($d_{ij}$ : matrice normalisée par $x/\\max$ pour un critère benefit, $\\min/x$ pour un coût ; $m$ : nombre d'alternatives.)

**Pourquoi l'utiliser ?** Aucun avis d'expert nécessaire, résultat **reproductible**.
*Limite* : ignore l'importance réelle des critères.
""")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    if not exiger_positif(X):
        return
    P, E, div, w = entropie_poids(X, cost)
    st.write("**Proportions $p_{ij}$**")
    aff(pd.DataFrame(P, index=alts, columns=crits))
    res = pd.DataFrame({"Entropie Ej": E, "1 - Ej": div, "Poids wj": w}, index=crits)
    aff(res)
    graphique_poids(crits, w)
    sauver_poids("Entropie", crits, w)


# ---------------------------------------------------------------- CRITIC
def page_critic():
    st.title("CRITIC")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode **objective**. Un critère est important s'il est **très dispersé** (écart-type élevé)
et **peu corrélé** avec les autres (il apporte une information nouvelle).

1. Normalisation min-max : benefit $r_{ij}=\\dfrac{x_{ij}-\\min x_j}{\\max x_j-\\min x_j}$ ; cost $r_{ij}=\\dfrac{\\max x_j-x_{ij}}{\\max x_j-\\min x_j}$
2. Coefficients de corrélation $\\rho_{jk}$ entre les colonnes
3. Indice d'information puis poids :
""")
        st.latex(r"C_j=\sigma_j\sum_{k=1}^{n}\left(1-|\rho_{jk}|\right) \qquad w_j=\frac{C_j}{\sum_j C_j}")
        st.markdown("""
**Pourquoi l'utiliser ?** Comme l'entropie, il n'exige aucun avis d'expert, mais il tient compte en plus
des **corrélations entre critères**.
""")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    R, rho, sigma, C, w = critic_poids(X, cost)
    st.write("**Matrice normalisée**")
    aff(pd.DataFrame(R, index=alts, columns=crits))
    st.write("**Corrélations $\\rho_{jk}$**")
    aff(pd.DataFrame(rho, index=crits, columns=crits))
    aff(pd.DataFrame({"Écart-type σj": sigma, "Indice Cj": C, "Poids wj": w}, index=crits))
    graphique_poids(crits, w)
    sauver_poids("CRITIC", crits, w)


# ---------------------------------------------------------------- WSM / WPM / WASPAS
def page_waspas():
    st.title("WSM - WPM - WASPAS")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthodes **additives / compensatoires** : un bon score sur un critère peut compenser un mauvais score
sur un autre. On normalise d'abord la matrice : benefit $r_{ij}=x_{ij}/\\max_i x_{ij}$ ; cost $r_{ij}=\\min_i x_{ij}/x_{ij}$.
""")
        st.latex(r"\textbf{WSM : } Q_i^{(1)}=\sum_{j}w_j\,r_{ij} \qquad \textbf{WPM : } Q_i^{(2)}=\prod_{j}(r_{ij})^{w_j}")
        st.latex(r"\textbf{WASPAS : } Q_i=\lambda\,Q_i^{(1)}+(1-\lambda)\,Q_i^{(2)},\quad \lambda\in[0,1]\ (\lambda=0{,}5 \Rightarrow Q_i=\tfrac12(Q^{(1)}_i+Q^{(2)}_i))")
        st.markdown("""
La meilleure alternative est celle qui a le **Q le plus élevé**.

**Pourquoi l'utiliser ?** Ce sont les méthodes les **plus simples** et faciles à expliquer.
WASPAS combine WSM et WPM pour une **meilleure précision**.
""")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    if not exiger_positif(X):
        return
    w = bloc_poids(crits, "wsm")
    lam = st.slider("λ (WASPAS)", 0.0, 1.0, 0.5, 0.05)
    R, q1, q2, q = wsm_wpm_waspas(X, cost, w, lam)
    st.write("**Matrice normalisée**")
    aff(pd.DataFrame(R, index=alts, columns=crits))
    out = pd.DataFrame({"WSM": q1, "WPM": q2, "WASPAS": q}, index=alts)
    for m in ["WSM", "WPM", "WASPAS"]:
        out["Rang " + m] = out[m].rank(ascending=False, method="min").astype(int)
    aff(out)
    st.success(f"Meilleure alternative (WASPAS) : **{out['WASPAS'].idxmax()}**")


# ---------------------------------------------------------------- TOPSIS
def page_topsis():
    st.title("TOPSIS")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** On construit une solution **idéale positive** $I^+$ (meilleur score de chaque critère) et une
solution **idéale négative** $I^-$ (pire score). On retient l'alternative **la plus proche de $I^+$** et
**la plus éloignée de $I^-$**.

1. Normalisation vectorielle : $r_{ij}=\\dfrac{x_{ij}}{\\sqrt{\\sum_i x_{ij}^2}}$
2. Matrice pondérée : $v_{ij}=w_j\\,r_{ij}$
3. Solutions idéales : $v_j^*=\\max_i v_{ij}$ (benefit) ou $\\min_i v_{ij}$ (cost) ; $v_j'$ l'inverse
4. Distances :
""")
        st.latex(r"S_i^+=\sqrt{\sum_j (v_j^*-v_{ij})^2} \qquad S_i^-=\sqrt{\sum_j (v_j'-v_{ij})^2}")
        st.markdown("5. Coefficient de proximité (le plus élevé = le meilleur) :")
        st.latex(r"RC_i^*=\frac{S_i^-}{S_i^+ + S_i^-}\in[0,1]")
        st.markdown("""
**Pourquoi l'utiliser ?** Très populaire : la logique de **distance à l'idéal** est intuitive et le calcul reste simple.
""")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    w = bloc_poids(crits, "top")
    R, V, ideal, anti, sp, sm, rc = topsis(X, cost, w)
    st.write("**Matrice normalisée pondérée $v_{ij}$**")
    aff(pd.DataFrame(V, index=alts, columns=crits))
    aff(pd.DataFrame({"Idéal positif I+": ideal, "Idéal négatif I-": anti}, index=crits))
    out = pd.DataFrame({"S+": sp, "S-": sm, "RC*": rc}, index=alts)
    out["Rang"] = out["RC*"].rank(ascending=False, method="min").astype(int)
    aff(out)
    meilleure(out, "RC*")


# ---------------------------------------------------------------- VIKOR
def page_vikor():
    st.title("VIKOR")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Méthode de **classement par compromis**. Pour chaque critère : $x_j^*$ = meilleure valeur et
$x_j^-$ = pire valeur.
""")
        st.latex(r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-} \qquad R_i=\max_j\left[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\right]")
        st.latex(r"Q_i=v\,\frac{S_i-S^*}{S^--S^*}+(1-v)\,\frac{R_i-R^*}{R^--R^*}")
        st.markdown("""
$S_i$ = utilité de **groupe**, $R_i$ = **regret individuel**, $v$ = poids de la stratégie de groupe (souvent 0,5).
$S^*=\\min S_i$, $S^-=\\max S_i$ (idem pour $R$). **Plus $Q$, $S$, $R$ sont petits, meilleure est l'alternative.**

**Solution de compromis** : l'alternative $A'$ (plus petit $Q$) est retenue si :
- **Avantage acceptable** : $Q(A'')-Q(A')\\ge \\dfrac{1}{m-1}$ ($A''$ = deuxième du classement Q)
- **Stabilité acceptable** : $A'$ est aussi la meilleure selon $S$ et $R$

Sinon : si seule la stabilité échoue, on propose $A'$ et $A''$ ; si l'avantage échoue, on propose $A', A'', ..., A_k$
avec $Q(A_k)-Q(A')<\\dfrac{1}{m-1}$.

**Pourquoi l'utiliser ?** Il maximise la **satisfaction du groupe** et minimise le **regret individuel** : utile
pour la **négociation** entre décideurs.
""")
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

    m = len(alts)
    ordre = out["Q"].sort_values().index.tolist()
    a1, a2 = ordre[0], ordre[1]
    c1 = out.loc[a2, "Q"] - out.loc[a1, "Q"] >= 1 / (m - 1)
    c2 = out.loc[a1, "Rang S"] == 1 and out.loc[a1, "Rang R"] == 1
    st.write(f"Condition 1 (avantage acceptable) : **{'vérifiée' if c1 else 'non vérifiée'}**")
    st.write(f"Condition 2 (stabilité acceptable) : **{'vérifiée' if c2 else 'non vérifiée'}**")
    if c1 and c2:
        st.success(f"Solution de compromis : **{a1}**")
    elif c1:
        st.info(f"Solutions de compromis : **{a1}** et **{a2}**")
    else:
        k = [a for a in ordre if out.loc[a, "Q"] - out.loc[a1, "Q"] < 1 / (m - 1)]
        st.info("Solutions de compromis : **" + "**, **".join(k) + "**")


# ---------------------------------------------------------------- Comparaison
def page_comparaison():
    st.title("Comparaison des classements (Spearman)")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** Aucune méthode n'est la meilleure dans l'absolu. On applique plusieurs méthodes avec les **mêmes poids**
et on mesure l'accord entre leurs classements avec la **corrélation de Spearman** :
""")
        st.latex(r"\rho=1-\frac{6\sum d_i^2}{n(n^2-1)}")
        st.markdown("""
$d_i$ = différence entre les deux rangs de l'alternative $i$.
$\\rho=+1$ : classements identiques ; $\\rho=0$ : aucun lien ; $\\rho=-1$ : classements inversés.

**Pourquoi l'utiliser ?** Pour vérifier la **robustesse** du choix : si les méthodes s'accordent sur les meilleures
alternatives, la décision est solide. Sinon, faire une analyse de sensibilité (poids, normalisation).
""")
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
    st.write("**Rang de chaque alternative (1 = la meilleure)**")
    st.dataframe(rangs)
    noms = list(rangs.columns)
    M = pd.DataFrame([[spearman(rangs[a], rangs[b]) for b in noms] for a in noms], index=noms, columns=noms)
    st.write("**Matrice de corrélation de Spearman**")
    aff(M, 2)


# ---------------------------------------------------------------- MAUT
def page_maut():
    st.title("Alternative unique (MAUT)")
    with st.expander("Principe, formules et pourquoi l'utiliser", expanded=True):
        st.markdown("""
**Principe.** On évalue **une seule alternative** de façon absolue (sans la comparer à d'autres). Chaque critère
est converti en **utilité** $u_j\\in[0,1]$, puis on agrège :
""")
        st.latex(r"S(A)=\sum_j w_j\,u_j(x_j)")
        st.latex(r"u_j=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}\ (\text{benefit}) \qquad u_j=\frac{x_j^{max}-x}{x_j^{max}-x_j^{min}}\ (\text{cost})")
        st.markdown("""
**Décision** avec un seuil $\\tau\\in[0,1]$ : **accepter si $S(A)\\ge\\tau$**, sinon rejeter.
Le seuil est fixé par des experts, la réglementation, ou estimé à partir de décisions passées (courbe ROC, indice de Youden).

**Pourquoi l'utiliser ?** Scoring de crédit, évaluation de performance, agrégation d'indicateurs (KPI) :
quand il n'y a **pas d'autres alternatives** à comparer.
""")
    d = donnees()
    if d is None:
        return
    alts, crits, X, cost = d
    w = bloc_poids(crits, "maut")
    st.write("**Valeurs de l'alternative à évaluer** (min et max : bornes du critère, par défaut tirées de la matrice)")
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
#  APPLICATION
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