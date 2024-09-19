import numpy as np
import matplotlib.pyplot as plt

# Esempio di array errore, dove i dati non sono un multiplo di 30
errore = np.random.rand(320)  # Dati di esempio per la variabile errore

# Numero di campioni per ogni boxplot (30 campioni per ogni boxplot)
n = 30

# Dividere i dati in gruppi di 30 campioni, senza scartare l'ultimo gruppo anche se ha meno di 30 elementi
gruppi = [errore[i:i + n] for i in range(0, len(errore), n)]

# Verifica la forma dei dati, devono essere una lista di array 1D
print([g.shape for g in gruppi])

# Creare il grafico con i boxplot successivi
plt.figure(figsize=(10, 6))
plt.boxplot(gruppi, positions=range(1, len(gruppi) + 1), widths=0.7)

# Aggiungere etichette e titoli
plt.xlabel('Gruppi di campioni (ogni gruppo contiene 30 campioni)')
plt.ylabel('Errore')
plt.title('Serie di Boxplot per l\'errore in funzione di b')

# Mostra il grafico
plt.show()
