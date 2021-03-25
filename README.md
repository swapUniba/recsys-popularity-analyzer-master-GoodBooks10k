# Framework per l'analisi del popularity bias
Il seguente framework permette l'esecuzione di diversi algoritmi di raccomandazione di tipo collaborative-filtering,
content-based e graph-based. Esso presenta anche script per il calcolo di metriche e generazioni di grafici a partire dall'output 
degli algoritmi di raccomandazione.

Presenta i moduli:

* collaborative-filtering: per l'esecuzione di algoritmi collaborativi offerti dalla libreria Lenskit 
* content-based-classification: per l'esecuzione di algoritmi content-based con classificazione (tecniche di apprendimento supervisionato libreria Scikit-learn)
* content-based-word-embedding: per l'esecuzione di algoritmi content-based con tecniche di word-embedding offerte dalla libreria Gensim
* pagerank e personalied_Pagerank: per l'esecuzione di algoritmi graph-based, ovvero Pagerank e Pagerank personalizzato (creazione del grafo e calcolo del pagerank libreria NetworkX)
* analysis: per il calcolo delle metriche/grafici 
* utils: per il mapping dei contenuti testuali, splitting degli utenti e altre operazioni di supporto

Potrebbe risultare utile scaricare la seguente cartella:
* datasets: https://mega.nz/folder/Lkg1UQ7a#LlhbJpK_I7zXqjF-A5MK3A (contenente i dati con cui le precedenti raccomandazioni sono state generate: GoodBooks-10k, tags, lista item popolari, suddivisione degli utenti)


## Installazione
* STEP1: assicurati di disporre di una versione python3.6 o superiore
* STEP2: apri il terminale e scrivi ```pip install numpy scipy pandas sklearn lenskit gensim matplotlib networkx```


## Come configurare ed eseguire gli algoritmi

### collaborative-filering
Per il lancio degli algoritmi cf e' semplicemente necessario lanciare lo script **run.py** all'interno di questo modulo. A differenza delle altre tecniche di raccomandazione, tale script esegue tutti gli algoritmi e serializza i diversi risultati in un file .parquet 



### content-based-classification e content-based-word-embedding
Questi due moduli presentano all'interno uno script chiamato **run.py** per l'esecuzione dei rispettivi algoritmi. Anche in questo caso, all'interno di tali script vi e' una sezione per la configurazione dei parametri con cui si vuole che gli algoritmi vengano lanciati.



### pagerank e personalied_Pagerank
Questi due moduli presentano all'interno uno script chiamato **run.py** per l'esecuzione dei rispettivi algoritmi. Anche in questo caso, all'interno di tali script vi e' una sezione per la configurazione dei parametri con cui si vuole che gli algoritmi vengano lanciati.



## Come eseguire il calcolo delle metriche
Il modulo analysis contiene lo script **run.py** per il calcolo di metriche e grafici a partire dai risultati prodotti da un determinato algoritmo content-based. Al suo interno vi e' una sezione per la configurazione, dove va riportato il nome dell'algoritmo, il percorso verso i risultati da esso prodotto, il path del dataset utilizzato. Per gli algoritmi di collaborative filtering e' necessario invece lanciare lo script **run_cf.py** vista la modalita' differente con cui Lenskit memorizza i risultati. 
