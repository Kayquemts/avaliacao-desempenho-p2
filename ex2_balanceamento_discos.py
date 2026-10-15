"""
Exercício 2 (Cap. 6) - Balanceamento do I/O das queries entre os dois discos

Enunciado (livro, Seção 6.8):
  "In the example of the transaction system of Sec. 6.2 it is noted that query
   transactions only make use of disk 1, which increases its utilization and
   turns it into the bottleneck. Having observed this problem, the support
   analyst wants to know what would be the effect on performance if the I/O
   load due to query transactions were balanced among the two disks. Compare
   the results obtained with the current situation."

Modelo: rede fechada de duas classes (update, query), N = (1, 3), resolvida com
o MVA exato multiclasse (Fig. 6.3 do livro).

Interpretação adotada (explicitada):  "balanceada entre os dois discos" =
a demanda de I/O das queries (0,180 s, hoje toda no Disco 1) é dividida em
partes iguais:  D(Disco1,query) = D(Disco2,query) = 0,090 s.  Isso é coerente
com a Seção 6.2 do livro (mesmos tempos de serviço nas duas classes; a demanda
muda apenas pelo nº de visitas) e com a divisão igualitária usada pelo livro
na alternativa "Upgrade in disk subsystem".

Cenários:
  A) Base ........... queries só no Disco 1       D_q = (0,105 ; 0,180 ; 0,000)
  B) Balanceado ..... queries 50% / 50%           D_q = (0,105 ; 0,090 ; 0,090)
  C) Todo no Disco 2  (cenário já resolvido no livro, usado só para VALIDAR o
     programa: X_q=4,335  X_u=0,517  R_q=0,692  R_u=1,934)
  + varredura da fração p da carga de I/O das queries enviada ao Disco 2.

Dependência: apenas numpy.     Execução:  python3 ex2_balanceamento_discos.py
"""
from itertools import product

import numpy as np

CLASSES = ["Update", "Query"]                # ordem do livro: (update, query)
DISPOSITIVOS = ["CPU", "Disco 1", "Disco 2"]
N = np.array([1, 3])                         # população (1 update, 3 query)
Z = np.zeros(2)

D_UPDATE = [0.375, 0.480, 0.240]
D_QUERY_CPU = 0.105
D_QUERY_IO = 0.180                           # demanda total de I/O das queries


def demandas(p):
    """p = fração do I/O das queries enviada ao Disco 2 (0 = base; 0,5 = balanceado)."""
    return np.array([D_UPDATE,
                     [D_QUERY_CPU, (1 - p) * D_QUERY_IO, p * D_QUERY_IO]])


# --------------------------------------------------------------------------
# MVA exato multiclasse (Fig. 6.3; Eqs. 6.1-6.5)
# --------------------------------------------------------------------------
def mva_exato(D, N, Z):
    R, K = D.shape
    n_tot = {tuple([0] * R): np.zeros(K)}
    for pop in product(*[range(n + 1) for n in N]):
        if sum(pop) == 0:
            continue
        Rp = np.zeros((R, K))
        X = np.zeros(R)
        for r in range(R):
            if pop[r] > 0:
                anterior = list(pop)
                anterior[r] -= 1                                  # N - 1_r
                Rp[r] = D[r] * (1.0 + n_tot[tuple(anterior)])     # Eq. 6.4/6.5
                X[r] = pop[r] / (Z[r] + Rp[r].sum())              # Eq. 6.1
        n_ri = X[:, None] * Rp                                    # Eq. 6.2
        n_tot[pop] = n_ri.sum(axis=0)                             # Eq. 6.3
    return {"X": X, "Rp": Rp, "n_ri": n_ri,
            "R": Rp.sum(axis=1),                       # R_r = sum_i R'_{i,r}
            "U_ri": X[:, None] * D,                    # U_{i,r} = X_r D_{i,r}
            "U_i": (X[:, None] * D).sum(axis=0)}


def resposta_media(res):
    """Tempo de resposta médio global (média das classes ponderada pela vazão)."""
    return float((res["X"] * res["R"]).sum() / res["X"].sum())


def verificar(nome, res, D):
    ok_little = np.allclose(res["X"] * res["R"], N, rtol=1e-9)        # N_r = X_r R_r
    ok_util = bool(np.all(res["U_i"] < 1.0))                          # U_i < 1
    ok_filas = np.allclose(res["n_ri"].sum(axis=1), N, atol=1e-9)     # soma das filas
    print(f"  [{nome}] Little N_r=X_r*R_r: {ok_little} | U_i<1: {ok_util} | "
          f"soma das filas = população: {ok_filas}")
    return ok_little and ok_util and ok_filas


def tabela_resultados(titulo, res, D):
    print(f"\n  {titulo}")
    print(f"  Demandas (s): Update = {D[0].round(4).tolist()} | Query = {D[1].round(4).tolist()}")
    print(f"  {'Classe':<8}{'X0 (tps)':>11}{'R (s)':>10}")
    for r, c in enumerate(CLASSES):
        print(f"  {c:<8}{res['X'][r]:>11.4f}{res['R'][r]:>10.4f}")
    print(f"  Resposta média global (ponderada por X0): {resposta_media(res):.4f} s")
    print(f"  {'Utilização (%)':<16}{'Update':>9}{'Query':>9}{'Total':>9}")
    for i, d in enumerate(DISPOSITIVOS):
        print(f"  {d:<16}{100*res['U_ri'][0, i]:>9.2f}{100*res['U_ri'][1, i]:>9.2f}"
              f"{100*res['U_i'][i]:>9.2f}")
    gargalo = DISPOSITIVOS[int(np.argmax(res["U_i"]))]
    print(f"  Gargalo (maior utilização): {gargalo} ({100*res['U_i'].max():.2f} %)")
    print(f"  {'Residência (%)':<16}{'Update':>9}{'Query':>9}      (R'_{{i,r}} / R_r x 100)")
    for i, d in enumerate(DISPOSITIVOS):
        print(f"  {d:<16}{100*res['Rp'][0, i]/res['R'][0]:>9.1f}"
              f"{100*res['Rp'][1, i]/res['R'][1]:>9.1f}")
    print(f"  {'Fila média n_i,r':<16}{'Update':>9}{'Query':>9}")
    for i, d in enumerate(DISPOSITIVOS):
        print(f"  {d:<16}{res['n_ri'][0, i]:>9.4f}{res['n_ri'][1, i]:>9.4f}")


def comparar(base, novo):
    print(f"\n  {'Medida':<28}{'Base':>10}{'Balanceado':>12}{'Variação %':>12}")
    def lin(nome, a, b):
        print(f"  {nome:<28}{a:>10.4f}{b:>12.4f}{100*(b-a)/a:>+12.2f}")
    for r, c in enumerate(CLASSES):
        lin(f"X0 {c} (tps)", base["X"][r], novo["X"][r])
    for r, c in enumerate(CLASSES):
        lin(f"R  {c} (s)", base["R"][r], novo["R"][r])
    lin("Resposta média global (s)", resposta_media(base), resposta_media(novo))
    lin("Vazão total (tps)", base["X"].sum(), novo["X"].sum())
    for i, d in enumerate(DISPOSITIVOS):
        lin(f"U {d} (%)", 100 * base["U_i"][i], 100 * novo["U_i"][i])


def main():
    print("=" * 74)
    print("EXERCÍCIO 2 - Balanceamento do I/O das queries entre os dois discos")
    print("=" * 74)

    base = mva_exato(demandas(0.0), N, Z)
    bal = mva_exato(demandas(0.5), N, Z)
    todo2 = mva_exato(demandas(1.0), N, Z)

    # ---- Validação contra o livro ----
    print("\n[0] VALIDAÇÃO contra o livro")
    print(f"  Base (Tab. 6.3):  X_u={base['X'][0]:.3f} (0,409) X_q={base['X'][1]:.3f} (4,093) "
          f"R_u={base['R'][0]:.3f} (2,444) R_q={base['R'][1]:.3f} (0,733)")
    print(f"  Util. base (Tab. 6.4): CPU={100*base['U_i'][0]:.2f} (58,31) "
          f"D1={100*base['U_i'][1]:.2f} (93,3) D2={100*base['U_i'][2]:.2f} (9,81)")
    print(f"  Cenário C (Seção 6.4, tudo no Disco 2): X_q={todo2['X'][1]:.3f} (4,335) "
          f"X_u={todo2['X'][0]:.3f} (0,517) R_q={todo2['R'][1]:.3f} (0,692) "
          f"R_u={todo2['R'][0]:.3f} (1,934)")

    # ---- Resultados ----
    print("\n" + "-" * 74)
    print("[1] SITUAÇÃO ATUAL (queries só no Disco 1)")
    print("-" * 74)
    tabela_resultados("Resultados - MVA exato, N=(1,3)", base, demandas(0.0))

    print("\n" + "-" * 74)
    print("[2] I/O DAS QUERIES BALANCEADO (50% Disco 1 / 50% Disco 2)")
    print("-" * 74)
    tabela_resultados("Resultados - MVA exato, N=(1,3)", bal, demandas(0.5))

    print("\n" + "-" * 74)
    print("[3] COMPARAÇÃO  balanceado x situação atual")
    print("-" * 74)
    comparar(base, bal)

    print("\n  Verificações de consistência:")
    verificar("base      ", base, demandas(0.0))
    verificar("balanceado", bal, demandas(0.5))

    # ---- Varredura complementar ----
    print("\n" + "-" * 74)
    print("[4] COMPLEMENTO: varredura da fração p do I/O das queries enviada ao Disco 2")
    print("-" * 74)
    print(f"  {'p':>5}{'X_upd':>9}{'X_qry':>9}{'R_upd':>9}{'R_qry':>9}{'R_médio':>9}"
          f"{'U_D1 %':>9}{'U_D2 %':>9}")
    melhor = None
    for p in np.round(np.arange(0.0, 1.0001, 0.1), 2):
        r = mva_exato(demandas(p), N, Z)
        rm = resposta_media(r)
        if melhor is None or rm < melhor[1]:
            melhor = (p, rm)
        print(f"  {p:>5.1f}{r['X'][0]:>9.4f}{r['X'][1]:>9.4f}{r['R'][0]:>9.4f}"
              f"{r['R'][1]:>9.4f}{rm:>9.4f}{100*r['U_i'][1]:>9.2f}{100*r['U_i'][2]:>9.2f}")
    print(f"  Menor resposta média global na grade testada: p = {melhor[0]:.1f} "
          f"(R_médio = {melhor[1]:.4f} s)")


if __name__ == "__main__":
    main()
