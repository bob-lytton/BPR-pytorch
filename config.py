# dataset name 
dataset = 'yelp_dataset'

# paths
main_path = os.path.realpath('~/datasets/yelp_dataset/rates')

# train_rating = main_path + '{}.train.rating'.format(dataset)
# test_rating = main_path + '{}.test.rating'.format(dataset)
# test_negative = main_path + '{}.test.negative'.format(dataset)
train_data = os.path.join(main_path, 'rate_train')
test_data = os.path.join(main_path, 'rate_test')
test_negative = os.path.join(main_path + 'test_with_neg')
user_data = os.path.join(main_path + 'num_to_userid')
item_data = os.path.join(main_path + 'num_to_businessid')

model_path = './models/'
# BPR_model_path = model_path + 'NeuMF.pth'
BPR_model_path = model_path + 'checkpoint.pth'
