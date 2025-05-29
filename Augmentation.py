## load dependencies
import smogn
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


data = pd.read_excel('Veri_Son.xlsx')
data = data.drop(["No"], axis=1)
data = data.drop(["Toplam"], axis=1)
data_ = data.sample(frac=1)

## conduct smogn
data_smogn = smogn.smoter(
    
    ## main arguments
    data = data_,           ## pandas dataframe
    y = 'Compressive Strength',          ## string ('header name')
    k = 7,                    ## positive integer (k < n)
    samp_method = 'extreme',  ## string ('balance' or 'extreme')

    ## phi relevance arguments
    rel_thres = 0.75,         ## positive real number (0 < R < 1)
    rel_method = 'auto',      ## string ('auto' or 'manual')
    rel_xtrm_type = 'high',   ## string ('low' or 'both' or 'high')
    rel_coef = 1.70           ## positive real number (0 < R)
)




data_smogn = smogn.smoter(
    
    ## main arguments
    data = data_smogn,           ## pandas dataframe
    y = 'Compressive Strength',          ## string ('header name')
    k = 7,                    ## positive integer (k < n)
    samp_method = 'extreme',  ## string ('balance' or 'extreme')

    ## phi relevance arguments
    rel_thres = 0.75,         ## positive real number (0 < R < 1)
    rel_method = 'auto',      ## string ('auto' or 'manual')
    rel_xtrm_type = 'high',   ## string ('low' or 'both' or 'high')
    rel_coef = 1.70           ## positive real number (0 < R)
)


print(data_smogn)




#seaborn.kdeplot(data_['Compressive Strength'], label = "Original")
#seaborn.kdeplot(data_smogn['Compressive Strength'], label = "Modified")



data_list = ["Compressive Strength", "Cement", "Silica Fume", "Quartz Sand", "Silica Sand", "Superplasticizer", "Micro Steel Fiber", "Water"]

fig, axes = plt.subplots(2, 4, figsize=(12, 8))  # Set figure size for better visualization

for i, data_name in enumerate(data_list):
  seaborn.kdeplot(data_[data_name], label="Original", ax=axes.flat[i])
  seaborn.kdeplot(data_smogn[data_name], label="Modified", ax=axes.flat[i])
  axes.flat[i].legend()
  

axes.flat[0].set_title("Compressive Strength")  # Set title for the first subplot
for ax, data_name in zip(axes.flat[1:], data_list[1:]):  # Set titles for remaining subplots (skipping the first)
    ax.set_title(data_name)
    ax.set_xlabel("")
    
plt.legend()  # Place legend in a better location
plt.tight_layout()  # Adjust spacing between subplots
plt.savefig("AugmentedData.png",format="png", dpi=300)
plt.show()

# Save augmented data
data_smogn.to_csv("AugmentedData.csv")

