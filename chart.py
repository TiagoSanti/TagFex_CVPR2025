import matplotlib.pyplot as plt
import numpy as np

# Dados dos experimentos - NME@1 (Nearest Mean Classifier)
classes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
icarl = [90.4, 78.15, 74.37, 68.82, 64.1, 60.03, 57.71, 53.79, 51.49, 49.11]
dytox = [92.4, 86.05, 79.13, 75.35, 74.0, 72.2, 68.49, 66.06, 64.09, 60.39]
der = [90.2, 80.3, 78.73, 75.48, 73.86, 71.47, 70.1, 66.94, 65.22, 64.35]
# TagFex baseline - NME@1 curve
tagfex = [93.20, 85.25, 82.73, 79.15, 75.82, 73.10, 70.56, 67.38, 64.76, 63.51]
# ANT β=0.5, m=0.5, Local - Melhor configuração (76.18% Avg NME@1)
ant = [93.20, 85.15, 83.43, 79.30, 76.22, 73.55, 71.66, 68.36, 66.08, 64.82]

# Criando o gráfico
plt.figure(figsize=(8, 6))

# Plotando as curvas com estilos, cores e símbolos
plt.plot(
    classes,
    icarl,
    marker="^",
    color="tab:blue",
    label="iCaRL",
    linestyle="-",
    markersize=8,
)
plt.plot(
    classes,
    dytox,
    marker="s",
    color="tab:cyan",
    label="DyTox",
    linestyle="-",
    markersize=8,
)
plt.plot(
    classes,
    der,
    marker="o",
    color="tab:green",
    label="DER",
    linestyle="-",
    markersize=8,
)
plt.plot(
    classes,
    tagfex,
    marker="*",
    color="tab:red",
    label="TagFex",
    linestyle="-",
    markersize=10,
)
plt.plot(
    classes,
    ant,
    marker="p",
    color="tab:purple",
    label="ANT",
    linestyle="-",
    markersize=10,
)

# Calculando e adicionando deltas do ANT em relação ao melhor dos outros
for i, class_num in enumerate(classes):
    # Pegar o melhor valor entre os outros métodos (exceto ANT)
    best_other = max(icarl[i], dytox[i], der[i], tagfex[i])
    delta = ant[i] - best_other
    sign = "+" if delta >= 0 else "-"
    delta = abs(delta)

    plt.text(
        class_num,
        ant[i] + 1.5,
        f"{sign} {delta:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        color="tab:purple",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="tab:purple",
            alpha=1,
        ),
    )

# Adicionando rótulos
plt.xlabel("Number of Classes", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)

# Definindo intervalos nos eixos X e Y
plt.xticks(np.arange(0, 101, 20))  # X-axis from 0 to 100 with increments of 20
plt.yticks(np.arange(50, 100, 10))  # Y-axis from 50 to 90 with increments of 10

# Adicionando a legenda
plt.legend()

# Exibindo a grade
plt.grid(True, which="both", linestyle="--", linewidth=1)

# Salvando o gráfico como uma imagem PNG
plt.savefig("cifar10_10.png", bbox_inches="tight", pad_inches=0.05)
