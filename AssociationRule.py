#Assignment on Association rule and Apriori Algorithm

data = [['Milk','Onion','Nutmeg','Kidney beans','Eggs','Yogurt'],
        ['Dill','Onion','Nutmeg','Kidney beans','Eggs','Yogurt'],
        ['Milk','Apple','Kidney beans','Eggs'],
        ['Milk','Unicorn','Corn','Kidney beans','Yogurt'],
        ['Corn','Onion','Onion','Kidney beans','Ice cream','Eggs']]

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

print(data)

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df,min_support=0.5, use_colnames=True)
print(frequent_itemsets)

from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(res)

res1 = res[['antecedents','consequents','support','confidence','lift']]
print(res1)

res2 = res1[res1['confidence'] >=1]
print(res2)