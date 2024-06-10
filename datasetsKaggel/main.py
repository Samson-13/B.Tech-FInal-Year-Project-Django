import kaggle

kaggle.api.authenticate()

# kaggle.api.dataset_download_files('datamunge/sign-language-mnist', path='./datasetsKaggel', unzip=True)
kaggle.api.dataset_download_files('ardamavi/sign-language-digits-dataset', path='./datasetsKaggel', unzip=True)

print(kaggle.api.dataset_list_files('datamunge/sign-language-mnist').files)



