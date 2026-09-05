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
