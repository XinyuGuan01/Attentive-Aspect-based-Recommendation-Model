# Attentive Aspect-based Recommendation Model

This is our implementation for the paper:

@article{guan2019attentive,
  title={Attentive Aspect Modeling for Review-aware Recommendation},
  author={Guan, Xinyu and Cheng, Zhiyong and He, Xiangnan and Zhang, Yongfeng and Zhu, Zhibo and Peng, Qinke and Chua, Tat-Seng},
  journal={ACM Transactions on Information Systems (TOIS)},
  volume={37},
  number={3},
  pages={28},
  year={2019},
  publisher={ACM}
}

**Please cite our TOIS paper if you use our codes.**

## Environment Settings
- tensorflow 1.2.1
- gensim 2.2.0

### Dataset
We provide the processed Amazon Beauty core-5 dataset. The original dataset can be found in [here](http://jmcauley.ucsd.edu/data/amazon/).

Beauty dataset.

users.txt:
- The users' names in the orignal dataset.
- Line `n` is the orginal name of user whose id is `n-1`

product.txt:
- The products' names in the orignal dataset.
- Line `n` is the orginal name of product whose id is `n-1`

aspect.txt:
- The aspects' names.
- We use *Sentires* to extract aspects from user reviews. The tool is available at [here](http://yongfeng.me/software/).
- Line `n` is the orginal name of aspect whose id is `n-1`

user_aspect_rank.txt:
- The aspect set of each user. 
- Line `n` is the aspect set of user whose id is `n-1`: aspect1,aspect2,...

item_aspect_rank.txt:
- The aspect set of each product. 
- Line `n` is the aspect set of product whose id is `n-1`: aspect1,aspect2,...

emb128.vector:
- The word embedding pretrained with Word2vec model (implemented with gensim). 
- Use `gensim.models.KeyedVectors.load_word2vec_format` to load the embedding matrix.

train_pairs.txt:
- Positive (user, item) pairs in training set.
- Each line is a training instance: userId,itemId

valid_pairs.txt
- Positive (user, item) pairs in validation set. 
- Each line is an instance: userId,itemId

test_pairs.txt:
- Positive (user, item) pairs in test set.
- Each line is a test instance: userId,itemId

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

Run aarm:
```
python running.py --productName Beauty --is_l2_regular 1 --lamda_l2 0.1 --is_out_l2 0 --dropout 0.5 --learning_rate 0.003 --num_aspect_factor 128 --num_mf_factor 128
```

