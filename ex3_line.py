"""
Exercício 3 (Cap. 6) - Modelo misto de três classes resolvido com a biblioteca LINE

Enunciado (livro, Seção 6.8, Tabela 6.12):
  "A computer system has a CPU and two disks, D1 and D2. The workload is divided
   into three classes: query (Q) transactions, update (U) transactions, and
   interactive (I) users. Table 6.12 gives the input parameters for these
   classes. Use QSolver/I to answer the following 'what if' questions:
   (a) What is the average response time for each class?
   (b) What is the impact on response time if the arrival rate of query
       transactions is increased by 95%?
   (c) In the scenario with an increased arrival rate of query transactions,
       consider the following hardware upgrades and compare the performance
       improvements obtained with each one of them:
         - replace disk D1 by one twice as fast.
         - replace the CPU by one twice as fast.
   (d) With the increased arrival rate for query transactions and with a
       twice-as-fast CPU, draw a graph of response time versus the number of
       terminals when the number of terminals varies from 50 to 250. What is
       the maximum number of terminals that can be supported to keep the
       response time for the interactive users below 1.5 sec?"

Tabela 6.12 (demandas em segundos):
  Classe   D_cpu   D_D1    D_D2   N_r^max  lambda_r (tps)  M_r   Z_r (s)
  Q        0,06    0,030   0,06     5         3,0           -     -
  U        0,10    0,030   0,09     3         1,5           -     -
  I        0,09    0,045   0,00     5          -           50    15

Modelagem na LINE:
  - Q e U: classes ABERTAS (OpenClass), chegadas Poisson (Source -> ... -> Sink).
  - I: classe FECHADA (ClosedClass) de M terminais, com tempo de reflexão Z em
    uma estação Delay (servidor de atraso / IS).
  - CPU, D1 e D2: estações Queue com disciplina PS (processor sharing). A PS é
    permitida pelo teorema BCMP com demandas distintas por classe, o que dá um
    modelo de forma produto (mesmas equações do livro, Fig. 6.9).
  - Cada classe visita cada dispositivo uma vez, com tempo de serviço = demanda
    D_{i,r}.  A classe I NÃO visita D2 (D = 0): seu roteamento é Delay -> CPU
    -> D1 -> Delay.
  - Solver: SolverMVA, método 'exact' (MVA exato; para modelos mistos a LINE
    resolve o submodelo aberto e o fechado em conjunto).
  - A coluna N_r^max (5, 3, 5) não é usada: o modelo é de forma produto, sem
    limite de multiprogramação / bloqueio por memória (o que o livro, Seção 6.7,
    declara explicitamente deixar para o Cap. 8).

Dependências:  pip install line-solver matplotlib numpy
Execução:      python3 ex3_line.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # gera arquivos PNG sem janela
import matplotlib.pyplot as plt

from line_solver import (Network, Source, Sink, Delay, Queue, OpenClass,
                         ClosedClass, SchedStrategy, Exp, SolverMVA,
                         GlobalConstants, VerboseLevel)

GlobalConstants.set_verbose(VerboseLevel.SILENT)   # silencia o log da LINE

# --------------------------------------------------------------------------
# Dados da Tabela 6.12
# --------------------------------------------------------------------------
CLASSES = ["Q", "U", "I"]
DEM = {  # demandas de serviço (s): (CPU, D1, D2)
    "Q": (0.06, 0.030, 0.06),
    "U": (0.10, 0.030, 0.09),
    "I": (0.09, 0.045, 0.00),
}
LAMBDA_Q, LAMBDA_U = 3.0, 1.5       # chegadas/s (classes abertas)
M_BASE, Z = 50, 15.0                # terminais e tempo de reflexão (classe I)
DISPOSITIVOS = ["CPU", "D1", "D2"]
LIMITE_R = 1.5                      # meta de tempo de resposta (item d)


# --------------------------------------------------------------------------
# Construção do modelo na LINE
# --------------------------------------------------------------------------
def construir_modelo(lam_q, lam_u, m_term, f_cpu=1.0, f_d1=1.0, z=Z):
    """f_cpu / f_d1 multiplicam as demandas da CPU / do disco D1
    (0,5 = dispositivo duas vezes mais rápido)."""
    modelo = Network("Exercicio3-misto")
    fonte = Source(modelo, "Source")
    sumidouro = Sink(modelo, "Sink")
    terminais = Delay(modelo, "Terminais")
    cpu = Queue(modelo, "CPU", SchedStrategy.PS)
    d1 = Queue(modelo, "D1", SchedStrategy.PS)
    d2 = Queue(modelo, "D2", SchedStrategy.PS)

    q = OpenClass(modelo, "Q")
    u = OpenClass(modelo, "U")
    i = ClosedClass(modelo, "I", m_term, terminais, 0)   # M terminais; ref.: Delay

    fonte.set_arrival(q, Exp(lam_q))
    fonte.set_arrival(u, Exp(lam_u))
    terminais.set_service(i, Exp.fit_mean(z))

    for nome, classe in (("Q", q), ("U", u), ("I", i)):
        d_cpu, d_d1, d_d2 = DEM[nome]
        cpu.set_service(classe, Exp.fit_mean(d_cpu * f_cpu))
        d1.set_service(classe, Exp.fit_mean(d_d1 * f_d1))
        if d_d2 > 0:
            d2.set_service(classe, Exp.fit_mean(d_d2))

    rota = modelo.init_routing_matrix()
    rota.set(q, q, Network.serial_routing(fonte, cpu, d1, d2, sumidouro))
    rota.set(u, u, Network.serial_routing(fonte, cpu, d1, d2, sumidouro))
    rota.set(i, i, Network.serial_routing(terminais, cpu, d1, terminais))
    modelo.link(rota)
    return modelo


# --------------------------------------------------------------------------
# Solução e extração de resultados
# --------------------------------------------------------------------------
def resolver(modelo, m_term):
    solver = SolverMVA(modelo, method="exact")
    nomes = modelo.get_station_names()                      # Source, Terminais, CPU, D1, D2
    idx = {n: k for k, n in enumerate(nomes)}
    fila = np.array(solver.avg_qlen())
    util = np.array(solver.avg_util())
    resp = np.array(solver.avg_respt())
    tput = np.array(solver.avg_tput())
    sys_resp = np.array(solver.avg_sys_resp_t())            # por cadeia: Q, U, I
    disp = [idx[d] for d in DISPOSITIVOS]

    res = {"X": {}, "R": {}, "R_sys": {}, "U_ri": {}, "n_ri": {}}
    for r, c in enumerate(CLASSES):
        res["X"][c] = tput[idx["CPU"], r]                   # vazão da classe
        res["R"][c] = resp[disp, r].sum()                   # soma dos tempos de resposta nos dispositivos
        res["R_sys"][c] = sys_resp[r] - (Z if c == "I" else 0.0)   # tempo de resposta do sistema (sem reflexão)
        res["U_ri"][c] = util[disp, r]
        res["n_ri"][c] = fila[disp, r]
    res["U_i"] = sum(res["U_ri"][c] for c in CLASSES)
    res["M"] = m_term
    return res


def resolver_cenario(lam_q=LAMBDA_Q, lam_u=LAMBDA_U, m_term=M_BASE, f_cpu=1.0, f_d1=1.0):
    return resolver(construir_modelo(lam_q, lam_u, m_term, f_cpu, f_d1), m_term)


# --------------------------------------------------------------------------
# Verificações de consistência com o enunciado / leis operacionais
# --------------------------------------------------------------------------
def verificar(nome, res, lam_q, lam_u, f_cpu=1.0, f_d1=1.0):
    lam = {"Q": lam_q, "U": lam_u}
    fator = [f_cpu, f_d1, 1.0]
    ok = True
    # 1) vazão das classes abertas = taxa de chegada (Eq. 6.11)
    c1 = all(abs(res["X"][c] - lam[c]) < 1e-9 for c in ("Q", "U"))
    # 2) lei da utilização U_{i,r} = X_r * D_{i,r}  (Eq. 6.15), com D da Tabela 6.12
    c2 = all(abs(res["U_ri"][c][k] - res["X"][c] * DEM[c][k] * fator[k]) < 1e-7
             for c in CLASSES for k in range(3))
    # 3) estabilidade: utilização total < 1 em todos os dispositivos (Eq. 6.23)
    c3 = bool(np.all(res["U_i"] < 1.0))
    # 4) lei de Little da classe fechada: M = X_I (Z + R_I)
    c4 = abs(res["X"]["I"] * (Z + res["R"]["I"]) - res["M"]) < 1e-6
    # 5) tempo de resposta somado nos dispositivos = tempo do sistema reportado pela LINE
    c5 = all(abs(res["R"][c] - res["R_sys"][c]) < 1e-9 for c in CLASSES)
    ok = c1 and c2 and c3 and c4 and c5
    print(f"  [{nome}] X_aberta=lambda: {c1} | U=X*D: {c2} | U_i<1: {c3} | "
          f"Little M=X(Z+R): {c4} | R(estações)=R(sistema): {c5}")
    return ok


# --------------------------------------------------------------------------
# Apresentação
# --------------------------------------------------------------------------
def mostrar(titulo, res):
    print(f"\n  {titulo}")
    print(f"  {'Classe':<8}{'X0 (tps)':>10}{'R (s)':>10}")
    for c in CLASSES:
        extra = f"   (R + Z = {res['R'][c] + Z:.4f} s)" if c == "I" else ""
        print(f"  {c:<8}{res['X'][c]:>10.4f}{res['R'][c]:>10.4f}{extra}")
    print(f"  {'Utilização %':<14}{'Q':>8}{'U':>8}{'I':>8}{'Total':>9}")
    for k, d in enumerate(DISPOSITIVOS):
        print(f"  {d:<14}{100*res['U_ri']['Q'][k]:>8.2f}{100*res['U_ri']['U'][k]:>8.2f}"
              f"{100*res['U_ri']['I'][k]:>8.2f}{100*res['U_i'][k]:>9.2f}")


def variacao(base, novo, titulo_base, titulo_novo):
    print(f"\n  {'Medida':<14}{titulo_base:>14}{titulo_novo:>14}{'Variação %':>12}")
    for c in CLASSES:
        a, b = base["R"][c], novo["R"][c]
        print(f"  R_{c} (s){'':<6}{a:>14.4f}{b:>14.4f}{100*(b-a)/a:>+12.2f}")
    a, b = base["X"]["I"], novo["X"]["I"]
    print(f"  X_I (tps){'':<5}{a:>14.4f}{b:>14.4f}{100*(b-a)/a:>+12.2f}")


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("EXERCÍCIO 3 - Modelo misto (Q, U abertas; I fechada) - biblioteca LINE")
    print("=" * 74)
    todos_ok = True

    # ---------------- (a) cenário base ----------------
    print("\n(a) CENÁRIO BASE: lambda_Q = 3,0 ; lambda_U = 1,5 ; M = 50 terminais ; Z = 15 s")
    base = resolver_cenario()
    mostrar("Resultados (LINE / SolverMVA exato)", base)
    todos_ok &= verificar("base", base, LAMBDA_Q, LAMBDA_U)

    # ---------------- (b) lambda_Q +95% ----------------
    lam_q2 = LAMBDA_Q * 1.95
    print(f"\n(b) lambda_Q AUMENTADA EM 95%: {LAMBDA_Q} -> {lam_q2:.2f} tps")
    b = resolver_cenario(lam_q=lam_q2)
    mostrar("Resultados (LINE / SolverMVA exato)", b)
    variacao(base, b, "Base (a)", "lambda_Q+95%")
    todos_ok &= verificar("b", b, lam_q2, LAMBDA_U)

    # ---------------- (c) upgrades ----------------
    print("\n(c) UPGRADES sobre o cenário (b)")
    c_d1 = resolver_cenario(lam_q=lam_q2, f_d1=0.5)
    mostrar("(c1) Disco D1 duas vezes mais rápido (D_D1 / 2)", c_d1)
    todos_ok &= verificar("c1", c_d1, lam_q2, LAMBDA_U, f_d1=0.5)
    c_cpu = resolver_cenario(lam_q=lam_q2, f_cpu=0.5)
    mostrar("(c2) CPU duas vezes mais rápida (D_CPU / 2)", c_cpu)
    todos_ok &= verificar("c2", c_cpu, lam_q2, LAMBDA_U, f_cpu=0.5)

    print("\n  Comparação dos upgrades em relação ao cenário (b) (variação % de R)")
    print(f"  {'Medida':<10}{'(b)':>10}{'D1 2x':>10}{'ganho %':>10}{'CPU 2x':>10}{'ganho %':>10}")
    for c in CLASSES:
        rb, r1, r2 = b["R"][c], c_d1["R"][c], c_cpu["R"][c]
        print(f"  R_{c} (s){'':<3}{rb:>10.4f}{r1:>10.4f}{100*(r1-rb)/rb:>+10.2f}"
              f"{r2:>10.4f}{100*(r2-rb)/rb:>+10.2f}")
    xb, x1, x2 = b["X"]["I"], c_d1["X"]["I"], c_cpu["X"]["I"]
    print(f"  X_I (tps){'':<1}{xb:>10.4f}{x1:>10.4f}{100*(x1-xb)/xb:>+10.2f}"
          f"{x2:>10.4f}{100*(x2-xb)/xb:>+10.2f}")

    # ---------------- (d) varredura do nº de terminais ----------------
    print("\n(d) lambda_Q = 5,85 ; CPU 2x mais rápida ; M variando de 50 a 250 terminais")
    ms = list(range(50, 251, 10))
    serie = {m: resolver_cenario(lam_q=lam_q2, f_cpu=0.5, m_term=m) for m in ms}
    print(f"  {'M':>5}{'R_I (s)':>10}{'X_I (tps)':>11}{'U_CPU %':>9}{'U_D1 %':>8}{'U_D2 %':>8}")
    for m in ms:
        r = serie[m]
        print(f"  {m:>5}{r['R']['I']:>10.4f}{r['X']['I']:>11.4f}{100*r['U_i'][0]:>9.2f}"
              f"{100*r['U_i'][1]:>8.2f}{100*r['U_i'][2]:>8.2f}")
        todos_ok &= (np.all(r["U_i"] < 1.0) and
                     abs(r["X"]["I"] * (Z + r["R"]["I"]) - m) < 1e-6)

    # maior M (inteiro) com R_I < 1,5 s: varre de 50 em diante (R_I é crescente em M)
    m_max, m = None, 50
    while m <= 2000:
        r = resolver_cenario(lam_q=lam_q2, f_cpu=0.5, m_term=m)
        if r["R"]["I"] >= LIMITE_R:
            break
        m_max = m
        m += 1
    r_ok = resolver_cenario(lam_q=lam_q2, f_cpu=0.5, m_term=m_max)
    r_nok = resolver_cenario(lam_q=lam_q2, f_cpu=0.5, m_term=m_max + 1)
    print(f"\n  Máximo de terminais com R_I < {LIMITE_R} s: M = {m_max}  "
          f"(R_I = {r_ok['R']['I']:.4f} s)")
    print(f"  Com M = {m_max + 1}: R_I = {r_nok['R']['I']:.4f} s (>= {LIMITE_R} s)")
    if m_max > 250:
        print("  OBS.: o limite está ACIMA do intervalo pedido (50 a 250): em todo o intervalo")
        print(f"        50-250 o requisito R_I < {LIMITE_R} s é atendido (R_I(250) = "
              f"{serie[250]['R']['I']:.4f} s).")

    # gráficos
    def grafico(arq, m_inicial, m_final, marcar_limite):
        xs = list(range(m_inicial, m_final + 1, 2))
        ys = [resolver_cenario(lam_q=lam_q2, f_cpu=0.5, m_term=x)["R"]["I"] for x in xs]
        plt.figure(figsize=(8, 4.8))
        plt.plot(xs, ys, lw=2, label="R_I (LINE, MVA exato)")
        plt.axhline(LIMITE_R, color="red", ls="--", label=f"Meta: {LIMITE_R} s")
        if marcar_limite:
            plt.axvline(m_max, color="green", ls=":", label=f"M máx. = {m_max}")
            plt.scatter([m_max], [r_ok["R"]["I"]], color="green", zorder=3)
        plt.xlabel("Número de terminais (M)")
        plt.ylabel("Tempo de resposta dos usuários interativos (s)")
        plt.title("Exercício 3(d): R_I x M (CPU 2x mais rápida, λ_Q = 5,85 tps)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(arq, dpi=150)
        plt.close()
        print(f"  Gráfico salvo em: {arq}")

    grafico("ex3_grafico_50_250.png", 50, 250, marcar_limite=False)
    grafico("ex3_grafico_estendido.png", 50, m_max + 30, marcar_limite=True)

    print("\n" + "=" * 74)
    print("Todas as verificações de consistência passaram:", bool(todos_ok))
    print("=" * 74)


if __name__ == "__main__":
    main()
