from Lupin.LupinExamin import LupinExamin

mobilenetExamin = LupinExamin(json='C:/Users/William/Documents/C3P0 datasets/greenscreen/testset/test.json', img_dir='C:/Users/William/Documents/C3P0 datasets/greenscreen/224x224_v1/',
                              kp_def='C:/Users/William/Documents/C3P0 datasets/greenscreen/dataset_head.csv', model_dir='C:/Users/William/Documents/git/master_models/quantium_models/cnn-train_split2')
mobilenetExamin.run_test()
# resnetExamin = LupinExamin()
# cnnExamin = LupinExamin()
# residualExamin = LupinExamin()
