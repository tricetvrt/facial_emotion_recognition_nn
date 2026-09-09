import matplotlib.pyplot as plt
import numpy as np

#resnet = [0.596, 0.585, 0.597, 0.344, 0.329, 0.304]
resnet2 = [0.71, 0.736, 0.686, 0.444, 0.52, 0.326]
mobilenet = [0.692, 0.738, 0.713, 0.381, 0.433, 0.378]
mobilevit = [0.646, 0.652, 0.63, 0.383, 0.397, 0.295]

x = np.arange(6)
width = 0.25  

plt.figure(figsize=(10, 6))

# Čuvamo povratne vrednosti (bar kontejnere) da bismo im dodali labele

bars2 = plt.bar(x, mobilenet, width, label="MobileNetV4 (small)", color="#ffc800")
bars3 = plt.bar(x + width, mobilevit, width, label="MobileViTV2", color="#ff7700")
bars1 = plt.bar(x - width, resnet2, width, label="ResNet-50 (LR=0.001)", color="#0011ff")
# Funkcija za ispisivanje vrednosti iznad barova
def dodaj_vrednosti(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Ispisujemo samo ako je vrednost veća od 0 (preskače nule kod MobileViT-a)
            plt.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 piksela vertikalni odmak iznad bara
                textcoords="offset points",
                ha='center', 
                va='bottom', 
                fontsize=8, 
                
            )

# Dodajemo vrednosti za svaki model
dodaj_vrednosti(bars1)
dodaj_vrednosti(bars2)
dodaj_vrednosti(bars3)

plt.ylim(0, 1.1)
plt.xlabel("Klasa", fontsize=11)
plt.ylabel("F1 score", fontsize=11)
plt.title("Poređenje F1 score-ova za svaku klasu po modelima")

plt.xticks(x, ["neutral", "happiness", "surprise", "sadness", "anger", "fear"])
plt.legend()
plt.tight_layout()
plt.show()