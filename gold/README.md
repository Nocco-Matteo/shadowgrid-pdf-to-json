# Gold set

## Protocollo di annotazione

1. Scegli 30–50 documenti rappresentativi dei **casi brutti**: pagine storte,
   timbri sovrapposti al testo, tabelle spezzate tra due pagine, scansioni
   sottoesposte, fotocopie di fotocopie.
2. Annota a mano **prima** di guardare l'output del modello. Se annoti dopo aver
   visto l'output, annoterai inconsciamente per compiacerlo.
3. Split: ~15 documenti di sviluppo (li guardi liberamente), il resto **sigillato**
   fino alla valutazione finale.

## Layout

- `gold/documents/` — PDF sorgenti.
- `gold/annotations/<doc_id>.json` — verità di terra (SchemaStrict).
- `gold/dev.txt` — lista doc_id di sviluppo.
- `gold/sealed.txt` — lista doc_id sigillati (non toccare fino alla valutazione).
- `gold/corrections/` — correzioni umane dalla Fase 7 (rientrano nel gold).

## Formato annotazione

JSON conforme a `SchemaStrict` (senza `quote`/`bbox`/`confidence` — solo i valori
attesi). La pipeline confronta `value` estratto vs `value` annotato.

## Stato

`annotations/doc_610757422887dfdbb907576eabbbf3a0.json` — 19 tratti razziali del
Player's Handbook (pp. 16-38), annotati dal **testo OCR** delle regioni, non dal
JSON prodotto dal modello.

**Riserva sul protocollo**: questa annotazione non rispetta il punto 2 qui sopra.
È stata scritta dopo aver letto l'output della run 3, quindi l'ancoraggio non è
zero — vale meno di un'annotazione alla cieca, e va trattata come tale.

Per questo `worksheet/<doc_id>.json` contiene lo stesso elenco con i valori da
compilare e il testo OCR di ogni tratto, ma **senza** le risposte: serve a fare
una seconda passata indipendente. Compilalo, poi confronta:

```bash
python -m pipeline.cli gold-diff --doc-id <doc_id>
```

Dove le due annotazioni divergono, o una delle due ha sbagliato o il caso è
genuinamente ambiguo — e un caso ambiguo nel gold set avvelena ogni misura che
ci fai sopra. Risolvi il disaccordo PRIMA di usarlo come metro.

## Identità degli elementi

Gli elementi di lista si annotano con `_anchor` e `_page`, non per posizione:
gli indici `traits[N]` cambiano da una run all'altra (`Damage Resistance` era
`traits[9]` nella run 1 e `traits[12]` nella run 3). La Fase 8 rimappa
`_anchor`+`_page` sull'indice reale leggendo l'inventario della Fase 4.
