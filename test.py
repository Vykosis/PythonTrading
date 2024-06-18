import pandas as pd
import pandas_ta as ta

df = pd.DataFrame({"number1":[1,2,3,4,5],"number2":[5,6,7,8,9]})
df.set_index("number1", inplace=True)
df = df.shift(1)
print(df)
  