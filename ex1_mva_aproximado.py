"""
Exercício 1 (Cap. 6) - Modelo fechado de duas classes (sistema de transações)

Enunciado (livro, Seção 6.8):
  "Consider the transaction system of Sec. 6.2. Calculate the performance
   measures of that model using the approximate MVA algorithm. Assume that the
   stopping criterion is to have a maximum difference of 0.001 for successive
   values of n_i,r. Suppose now that the multiprogramming level of the update
   class is tripled. Recalculate the model's results using the exact and
   approximate techniques. Compare the computational effort required by the
   two algorithms."

Dados (Tabela 6.1 / Fig. 6.1) - demandas de serviço em segundos:
              CPU     Disco 1   Disco 2   MPL
   Update    0,375     0,480     0,240      1
   Query     0,105     0,180     0,000      3

Algoritmos implementados (todos do próprio Cap. 6):
  - MVA exato multiclasse ........ Fig. 6.3 (Eqs. 6.1, 6.2, 6.3, 6.4, 6.5)
  - MVA aproximado (Schweitzer) .. Fig. 6.5 (Eqs. 6.9 e 6.10)

Dependência: apenas numpy.     Execução:  python3 ex1_mva_aproximado.py
"""
import time
from itertools import product

import numpy as np

# --------------------------------------------------------------------------
# Dados do problema (a ordem das classes segue o livro: (update, query))
# --------------------------------------------------------------------------
CLASSES = ["Update", "Query"]
DISPOSITIVOS = ["CPU", "Disco 1", "Disco 2"]
D = np.array([[0.375, 0.480, 0.240],     # Update: D[r, i]
              [0.105, 0.180, 0.000]])    # Query
Z = np.zeros(2)                          # sem terminais (classes "batch")
N_BASE = np.array([1, 3])                # população do modelo base (Tab. 6.1)
N_TRIPLO = np.array([3, 3])              # MPL da classe update triplicado
EPS = 0.001                              # critério de parada do enunciado


# --------------------------------------------------------------------------
# MVA exato multiclasse (Fig. 6.3)
# --------------------------------------------------------------------------
def mva_exato(D, N, Z):
    """Retorna dict com X (vazão por classe), Rp (R'_{r,i}), n_ri (fila por
    classe/dispositivo) e 'ops' = nº de cálculos de tempo de residência."""
    R, K = D.shape
    n_tot = {tuple([0] * R): np.zeros(K)}      # n_i(populacao)
    ops = 0
    for pop in product(*[range(n + 1) for n in N]):   # ordem lexicográfica
        if sum(pop) == 0:
            continue
        Rp = np.zeros((R, K))
        X = np.zeros(R)
        for r in range(R):
            if pop[r] > 0:
                anterior = list(pop)
                anterior[r] -= 1                       # N - 1_r  (Eq. 6.5)
                Rp[r] = D[r] * (1.0 + n_tot[tuple(anterior)])   # Eq. 6.4
                ops += K
                X[r] = pop[r] / (Z[r] + Rp[r].sum())            # Eq. 6.1
        n_ri = X[:, None] * Rp                                  # Eq. 6.2
        n_tot[pop] = n_ri.sum(axis=0)                           # Eq. 6.3
    return {"X": X, "Rp": Rp, "n_ri": n_ri, "ops": ops, "iter": None}


# --------------------------------------------------------------------------
# MVA aproximado de Schweitzer (Fig. 6.5)
# --------------------------------------------------------------------------
def mva_aproximado(D, N, Z, eps, max_iter=100000):
    """Critério de parada: máx |n_novo - n_antigo| < eps (diferença absoluta
    entre valores sucessivos de n_{i,r}, como pede o enunciado)."""
    R, K = D.shape
    Kr = (D > 0).sum(axis=1)                         # nº de dispositivos com D>0
    n_ri = np.where(D > 0, N[:, None] / Kr[:, None], 0.0)   # estimativa inicial
    it = 0
    while it < max_iter:
        it += 1
        antigo = n_ri.copy()
        n_i = antigo.sum(axis=0)
        Rp = np.zeros((R, K))
        X = np.zeros(R)
        for r in range(R):
            # fila vista por um cliente da classe r:  n_i(N - 1_r)  (Eq. 6.10)
            #   = sum_{t != r} n_{i,t}(N) + (N_r - 1)/N_r * n_{i,r}(N)
            vista = n_i - antigo[r] / N[r]
            Rp[r] = D[r] * (1.0 + vista)                      # Eq. 6.4
            X[r] = N[r] / (Z[r] + Rp[r].sum())                # Eq. 6.1
        n_ri = X[:, None] * Rp                                # Eq. 6.2
        if np.max(np.abs(n_ri - antigo)) < eps:
            break
    return {"X": X, "Rp": Rp, "n_ri": n_ri, "ops": it * R * K, "iter": it}


# --------------------------------------------------------------------------
# Métricas derivadas e verificações
# --------------------------------------------------------------------------
def metricas(res, D, N, Z):
    R_cls = res["Rp"].sum(axis=1)                 # R_r = soma dos R'_{i,r}
    U_ri = res["X"][:, None] * D                  # U_{i,r} = X_r * D_{i,r}
    return {"R": R_cls, "U_ri": U_ri, "U_i": U_ri.sum(axis=0)}


def verificar(nome, res, D, N, Z):
    m = metricas(res, D, N, Z)
    tol = 5e-3 if res["iter"] is not None else 1e-9   # aproximado: tolerância maior
    ok_little = np.allclose(res["X"] * (Z + m["R"]), N, rtol=1e-9, atol=1e-9)
    ok_util = bool(np.all(m["U_i"] < 1.0))
    ok_pop = np.allclose(res["n_ri"].sum(axis=1), N, atol=tol * N.max())
    print(f"  [{nome}] Little X_r*(Z+R_r)=N_r: {ok_little} | "
          f"U_i<1: {ok_util} | soma das filas = população: {ok_pop}")
    return ok_little and ok_util and ok_pop


def imprimir(titulo, res, D, N, Z):
    m = metricas(res, D, N, Z)
    print(f"\n  {titulo}")
    print(f"  {'Classe':<8}{'N_r':>4}{'X0 (tps)':>11}{'R (s)':>10}")
    for r, c in enumerate(CLASSES):
        print(f"  {c:<8}{N[r]:>4}{res['X'][r]:>11.4f}{m['R'][r]:>10.4f}")
    print(f"  {'Disp.':<8}{'U Update':>10}{'U Query':>10}{'U total':>10}"
          f"{'n Update':>10}{'n Query':>10}{'n total':>10}")
    for i, d in enumerate(DISPOSITIVOS):
        print(f"  {d:<8}{m['U_ri'][0, i]:>10.2%}{m['U_ri'][1, i]:>10.2%}"
              f"{m['U_i'][i]:>10.2%}{res['n_ri'][0, i]:>10.4f}"
              f"{res['n_ri'][1, i]:>10.4f}{res['n_ri'][:, i].sum():>10.4f}")


def erro_relativo(aprox, exato):
    return 100.0 * np.abs(aprox - exato) / exato


def comparar(res_a, res_e, D, N, Z):
    ma, me = metricas(res_a, D, N, Z), metricas(res_e, D, N, Z)
    print(f"\n  Erro relativo (|%|) aproximado x exato")
    print(f"  {'Classe':<8}{'X0':>10}{'R':>10}")
    for r, c in enumerate(CLASSES):
        print(f"  {c:<8}{erro_relativo(res_a['X'][r], res_e['X'][r]):>10.3f}"
              f"{erro_relativo(ma['R'][r], me['R'][r]):>10.3f}")
    print(f"  Maior erro relativo (X0 e R): "
          f"{max(erro_relativo(res_a['X'], res_e['X']).max(), erro_relativo(ma['R'], me['R']).max()):.3f} %")


def tempo_medio(func, repeticoes=2000):
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        func()
    return (time.perf_counter() - t0) / repeticoes * 1e6   # microssegundos


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    K, R = D.shape[1], D.shape[0]
    print("=" * 74)
    print("EXERCÍCIO 1 - Modelo de duas classes: MVA aproximado x MVA exato")
    print("=" * 74)

    # ---- Validação das implementações contra valores publicados no livro ----
    print("\n[0] VALIDAÇÃO contra o livro (Tabelas 6.3 e 6.6)")
    ex = mva_exato(D, N_BASE, Z)
    me = metricas(ex, D, N_BASE, Z)
    print(f"  Exato, N=(1,3): X_update={ex['X'][0]:.3f} (livro 0,409) | "
          f"X_query={ex['X'][1]:.3f} (livro 4,093) | "
          f"R_update={me['R'][0]:.3f} (livro 2,444) | R_query={me['R'][1]:.3f} (livro 0,733)")
    ap01 = mva_aproximado(D, N_BASE, Z, 0.01)
    ma01 = metricas(ap01, D, N_BASE, Z)
    print(f"  Aproximado, eps=0,01: X_query={ap01['X'][1]:.3f} (livro 4,001) | "
          f"X_update={ap01['X'][0]:.3f} (livro 0,407) | "
          f"R_query={ma01['R'][1]:.3f} (livro 0,749) | R_update={ma01['R'][0]:.3f} (livro 2,456)")

    # ---- Parte 1: modelo base, MVA aproximado com eps = 0,001 ----
    print("\n" + "-" * 74)
    print(f"[1] MODELO BASE  N = (1 update, 3 query)   |   eps = {EPS}")
    print("-" * 74)
    ap_base = mva_aproximado(D, N_BASE, Z, EPS)
    imprimir(f"MVA APROXIMADO (Schweitzer) - convergiu em {ap_base['iter']} iterações",
             ap_base, D, N_BASE, Z)
    imprimir("MVA EXATO (referência)", ex, D, N_BASE, Z)
    comparar(ap_base, ex, D, N_BASE, Z)
    print("\n  Verificações de consistência:")
    verificar("aprox. base", ap_base, D, N_BASE, Z)
    verificar("exato  base", ex, D, N_BASE, Z)

    # ---- Parte 2: MPL da classe update triplicado -> N = (3,3) ----
    print("\n" + "-" * 74)
    print(f"[2] MPL DA CLASSE UPDATE TRIPLICADO  N = (3 update, 3 query)   |   eps = {EPS}")
    print("-" * 74)
    ex3 = mva_exato(D, N_TRIPLO, Z)
    ap3 = mva_aproximado(D, N_TRIPLO, Z, EPS)
    imprimir("MVA EXATO", ex3, D, N_TRIPLO, Z)
    imprimir(f"MVA APROXIMADO - convergiu em {ap3['iter']} iterações", ap3, D, N_TRIPLO, Z)
    comparar(ap3, ex3, D, N_TRIPLO, Z)
    print("\n  Verificações de consistência:")
    verificar("exato  (3,3)", ex3, D, N_TRIPLO, Z)
    verificar("aprox. (3,3)", ap3, D, N_TRIPLO, Z)

    # variação em relação ao modelo base (resultados exatos)
    m3 = metricas(ex3, D, N_TRIPLO, Z)
    print("\n  Impacto de triplicar o MPL de update (resultados EXATOS, (3,3) x (1,3))")
    print(f"  {'Classe':<8}{'X0 base':>10}{'X0 novo':>10}{'var %':>9}{'R base':>10}{'R novo':>10}{'var %':>9}")
    for r, c in enumerate(CLASSES):
        xb, xn, rb, rn = ex["X"][r], ex3["X"][r], me["R"][r], m3["R"][r]
        print(f"  {c:<8}{xb:>10.4f}{xn:>10.4f}{100*(xn-xb)/xb:>+9.2f}"
              f"{rb:>10.4f}{rn:>10.4f}{100*(rn-rb)/rb:>+9.2f}")

    # ---- Parte 3: esforço computacional ----
    print("\n" + "-" * 74)
    print("[3] COMPARAÇÃO DO ESFORÇO COMPUTACIONAL")
    print("-" * 74)
    prod_pop = lambda N: int(np.prod(N + 1))
    linhas = [("Base (1,3)", N_BASE, ex, ap_base), ("Triplicado (3,3)", N_TRIPLO, ex3, ap3)]
    print(f"  {'Cenário':<18}{'Populações':>11}{'K·R·Π(1+N_r)':>14}{'Cálc. R\' exato':>16}"
          f"{'Iter. aprox.':>14}{'K·R·iter':>10}")
    for nome, N, e, a in linhas:
        print(f"  {nome:<18}{prod_pop(N):>11}{K * R * prod_pop(N):>14}{e['ops']:>16}"
              f"{a['iter']:>14}{a['ops']:>10}")
    print("  (K·R·Π(1+N_r) é a Eq. 6.6; 'Cálc. R'' = nº de vezes que o tempo de residência")
    print("   R'_{i,r} foi efetivamente calculado; K·R·iter = custo do aproximado.)")
    print(f"\n  Tempo médio de CPU por execução (apenas ilustrativo - modelo minúsculo):")
    for nome, N, e, a in linhas:
        te = tempo_medio(lambda: mva_exato(D, N, Z))
        ta = tempo_medio(lambda: mva_aproximado(D, N, Z, EPS))
        print(f"  {nome:<18} exato = {te:8.1f} µs | aproximado = {ta:8.1f} µs")
    print("\n  Observação: o custo do exato cresce com o produto Π(1+N_r) (exponencial no")
    print("  nº de classes); o do aproximado cresce com K·R vezes o nº de iterações.")

    # Extrapolação (exemplo hipotético da Seção 6.4.1: 20 classes, 5 dispositivos,
    # 1 cliente por classe) - só aplica a Eq. 6.6; não executa o algoritmo.
    K_h, R_h = 5, 20
    custo_exato = K_h * R_h * 2 ** R_h
    print(f"\n  Extrapolação (R={R_h} classes, K={K_h} dispositivos, N_r=1):")
    print(f"    exato     : K·R·Π(1+N_r) = {custo_exato:,} operações")
    print(f"    aproximado: K·R = {K_h * R_h} operações por iteração "
          f"(~{10 * K_h * R_h:,} para ~10 iterações)")
    print("  Conclusão: para o modelo pequeno deste exercício o exato é tão barato quanto o")
    print("  aproximado (ou mais); a vantagem do aproximado só aparece com muitas classes.")


if __name__ == "__main__":
    main()
