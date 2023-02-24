
### mobilePriceRange 
itemsets, rules = apriori(transactions, min_support=0.13, min_confidence=0.24, output_transaction_ids=False)

```
rule:{('four_g', 1)} -> {('price_range', 3)} (conf: 0.264, supp: 0.138, lift: 1.055, conv: 1.019)
rule:{('ram', 'very low')} -> {('price_range', 0)} (conf: 0.917, supp: 0.183, lift: 3.670, conv: 9.091)
rule:{('touch_screen', 0)} -> {('price_range', 2)} (conf: 0.267, supp: 0.133, lift: 1.066, conv: 1.023)
rule:{('ram', 'very high')} -> {('price_range', 3)} (conf: 0.887, supp: 0.177, lift: 3.550, conv: 6.667)
rule:{('four_g', 1), ('three_g', 1)} -> {('price_range', 3)} (conf: 0.264, supp: 0.138, lift: 1.055, conv: 1.019)
rule:{('ram', 'very low'), ('three_g', 1)} -> {('price_range', 0)} (conf: 0.911, supp: 0.138, lift: 3.642, conv: 8.389)
rule:{('ram', 'very high'), ('three_g', 1)} -> {('price_range', 3)} (conf: 0.887, supp: 0.137, lift: 3.547, conv: 6.621)
```