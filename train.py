import time, pickle
from model5 import *
import numpy as np
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='3'

def GetBatch(data, batch_size, data_size):
    idx = np.arange(0, data_size)
    permutation = np.random.permutation(idx)
    pickIdx = permutation[:batch_size]
    d, q, a = [], [], []
    for idx in pickIdx:
        d.append(data[idx][0])
        q.append(data[idx][1])
        a.append(data[idx][2])
    return np.array(d), np.array(q), np.array(a)

def Accuracy(si, d, a):
    correct = 0
    for sampleId in range(d.shape[0]):
        prob = defaultdict(float)
        for idx, wordId in enumerate(d[sampleId]):
            prob[wordId] += si[sampleId][idx]
        predict = max(prob, key=prob.get)
        # print("Sum si: {}, predict id: {}, prob: {}, true id: {}, true id prob: {}".format(sum(si[sampleId]), predict, prob[predict], a[sampleId], prob[a[sampleId]]))
        if predict == a[sampleId]:
            correct += 1
    accuracy = correct / d.shape[0]
    # print('#samples in !',d.shape[0])
    return accuracy, correct

def CheckBatch(data, batch_size):
    d, q, a = [], [], []
    for i in range(batch_size):
        d.append(data[i][0])
        q.append(data[i][1])
        a.append(data[i][2])
    return np.array(d), np.array(q), np.array(a)


def main():
    vocab_size, vocab, word2idx, idx2word, dl, ql, cl = pickle.load(open(vocab_path, 'rb'))
    print("Load data!")
    train_data = pickle.load(open(train_idx_path, 'rb'))
    valid_data = pickle.load(open(valid_idx_path, 'rb'))
    print("Done!\nBuild model!")
    model = Reader(embedding_dim=384, hidden_dim=384, graph_dim=256, dLen=dl, qLen=ql, learning_rate=learning_rate)
    model.build()
    print("Done!")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs_q4h_384/", sess.graph)

        ckpt = tf.train.latest_checkpoint('./')
        if ckpt != None:
            saver.restore(sess, ckpt)
        else:
            print("No pre-trained model!")

        t = time.time()
        print("Start training the model!")
        for epoch in range(2,4):
            for step in range(TRAIN_SIZE // BATCH_SIZE):
                Bd, Bq, Ba = GetBatch(train_data, BATCH_SIZE, TRAIN_SIZE)
                lossg, tloss, _, tsi = sess.run([merged, model.loss, model.opt, model.si],
                                       feed_dict={model.placeholders['d']: Bd,
                                                  model.placeholders['q']: Bq,
                                                  model.placeholders['a']: Ba,
                                                  model.placeholders['k']: 0.4})
                accuT = Accuracy(tsi, Bd, Ba)
                writer.add_summary(lossg, step)

                if step % 10 == 0:
                    Bdv, Bqv, Bav = GetBatch(valid_data, batch_size=48, data_size=VALID_SIZE)
                    vsi, vloss = sess.run([model.si, model.loss],
                                          feed_dict={model.placeholders['d']: Bdv,
                                                     model.placeholders['q']: Bqv,
                                                     model.placeholders['a']: Bav,
                                                     model.placeholders['k']: 1})
                    accuV = Accuracy(vsi, Bdv, Bav)
                    print("EPOCH {} - step {}: Training loss: {:.2f}, Training accuracy: {:.2f},\n Valid loss: {:.2f}, Valid accuracy: {:.2f} cost {:.2f}".format(epoch, step, tloss, accuT, vloss, accuV, time.time()-t))
                    if step % 100 == 0:
                        saver.save(sess, './reader.ckpt', global_step=epoch+step)
                    t = time.time()

def GetTest(data, i, batch_size):
    d, q, a = [], [], []
    if i <= 61:
        for j in range(i*batch_size, (i+1)*batch_size):
            d.append(data[j][0])
            q.append(data[j][1])
            a.append(data[j][2])
    else:
        for j in range(1984, 2000):
            d.append(data[j][0])
            q.append(data[j][1])
            a.append(data[j][2])
    return np.array(d), np.array(q), np.array(a)

def test():
    vocab_size, vocab, word2idx, idx2word, dl, ql, cl = pickle.load(open(vocab_path, 'rb'))
    # train_data = pickle.load(open(train_idx_path, 'rb'))
    valid_data = pickle.load(open(valid_idx_path, 'rb'))
    model = Reader(embedding_dim=384, hidden_dim=384, graph_dim=384, dLen=dl, qLen=ql, learning_rate=learning_rate)
    model.build()
    print("Model buliding finished!!\nStart testing!!!")

    t = time.time()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt != None:
            print("Find pretrained model!")
            saver.restore(sess, './reader.ckpt')
            accu = []
            allcorrect = 0
            for i in range(63):
                Bdv, Bqv, Bav = GetTest(valid_data, batch_size=32, i=i)
                vsi, valid_loss = sess.run([model.si, model.loss],
                                           feed_dict={model.placeholders['d']: Bdv, model.placeholders['q']: Bqv,
                                                      model.placeholders['a']: Bav,
                                                      model.placeholders['k']: 1})
                accuV, correct = Accuracy(vsi, Bdv, Bav)
                allcorrect += correct
                accu.append(accuV)
                print("{}th DATASET, Valid loss: {}, Valid accuracy: {}".format(i + 1, valid_loss, accuV))
            print('Total accu', accu)
            print('Allcorrect #', allcorrect)
        else:
            print("Stop the program!")
if __name__ == '__main__':
    # main()
    test()
