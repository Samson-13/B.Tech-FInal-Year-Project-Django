import kaggle

kaggle.api.authenticate()

# kaggle.api.dataset_download_files('datamunge/sign-language-mnist', path='./datasetsKaggel', unzip=True)



# kaggle.api.dataset_metadata('datamunge/sign-language-mnist', path='./datasetsKaggel')


print(kaggle.api.dataset_list_files('datamunge/sign-language-mnist').files)



