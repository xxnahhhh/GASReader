"""idea 1： 用GCN--> embedding e encoder f g
(1)利用 entity: fi(d)fj(d)内积,选出topm个neighbor 构造adjacency matrix --> G(X,V,E)
(2)用GCN提取node结构信息,(默认candidates在documents里面出现过):
    1'Attentive Sum Reader: node representations: GCN(fi(d))(entity)和q的内积过softmax加和最大
    2'softmax(fi(d)*q).dot(candidate)(candidates are also entity)-->cross entropy
    
    idea 2:试一下link prediction
# """
import tensorflow as tf
from config import *
import pickle

class Reader(object):
    def __init__(self,embedding_dim, hidden_dim, graph_dim, dLen, qLen, learning_rate, **kwargs):
        allowed_keys = {'vocab_size'}
        for k in kwargs:
            assert k in allowed_keys, "Invalid keyword argument: "+ k

        vocab_size = kwargs.get('vocab_size')
        if not vocab_size:
            vocab_size = VOCAB_SIZE  # should be refined to VOCAB_SIZE

        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.hidden_dim = hidden_dim
        self.dLen = dLen
        self.qLen = qLen
        self.learning_rate = learning_rate
        self.graph_dim = graph_dim
        self.vars = {}


    def BuildP(self):
        d = tf.placeholder(dtype=tf.int32, shape=[None, self.dLen], name='document')
        q = tf.placeholder(dtype=tf.int32, shape=[None, self.qLen], name='query')
        a = tf.placeholder(dtype=tf.int32, shape=[None], name='answer')
        keep_prob = tf.placeholder(tf.float32)

        placeholders = {'d': d, 'q': q, 'a': a, 'k': keep_prob}

        return placeholders

    def predict(self):
        with tf.variable_scope('Reader'):
            # self.embeddings = tf.Variable(pickle.load(open('./pretrained_embeddings.pkl', 'rb')), name='embeddings', dtype=tf.float32,trainable=False)
            embeddings = tf.Variable(pickle.load(open('./pretrained_embeddings.pkl', 'rb')), name='embeddings', dtype=tf.float32)

        with tf.variable_scope('f'):
            d = self.placeholders['d']
            embedD = tf.nn.embedding_lookup(params=embeddings, ids=d)
            seqdLen = tf.reduce_sum(tf.cast(tf.not_equal(d, 0), tf.int32), 1)
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
            hi, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,inputs=embedD, sequence_length=seqdLen,dtype=tf.float32)

            self.fid = tf.concat(hi, 2)
        with tf.variable_scope('g'):
            q = self.placeholders['q']
            embedQ = tf.nn.embedding_lookup(params=embeddings, ids=q)
            seqqLen = tf.reduce_sum(tf.cast(tf.not_equal(q, 0), tf.int32), 1)
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
            # q,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,inputs=embedQ, sequence_length=seqqLen,dtype=tf.float32)
            _, q = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,inputs=embedQ, sequence_length=seqqLen,dtype=tf.float32)

            self.qi = tf.concat(q, 1)

        with tf.variable_scope('GCN'):
            adj = self.topm(self.fid, self.qi, m=50)
            # fg, qg = self.GCN(adj, self.graph_dim)
            self.fg1 = self.GCN(adj,self.fid,1, shape=[2*self.hidden_dim,self.graph_dim])
            self.fg = self.GCN(adj,self.fg1,2, shape=[self.graph_dim, 2*self.hidden_dim])

        with tf.variable_scope('AttentiveSum'):
            # ff = tf.concat([fid, fg], axis=2)
            # qq = tf.reduce_sum(tf.concat([qi, qg], axis=2), axis=1)
            # logit = tf.reshape(tf.matmul(ff, tf.expand_dims(qq, -1)), shape=[-1, self.dLen])
            d = self.placeholders['d']
            self.ffg = tf.expand_dims(tf.cast(tf.not_equal(d, 0), tf.float32), -1) * self.fg
            self.ff = tf.concat([self.fid, self.ffg], axis=2)

            w_q = tf.get_variable(name='w_q', shape=[2*self.hidden_dim, 2*self.hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='q_b', shape=[2*self.hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
            newq = tf.nn.relu(tf.matmul(self.qi, w_q) + b)
            newq = tf.nn.dropout(newq, keep_prob=self.placeholders['k'])
            newq2 = tf.concat([self.qi, newq], axis=1)
            self.logit1 = tf.reshape(tf.matmul(self.ff, tf.expand_dims(newq2, -1)), shape=[-1, self.dLen])
            logit2 = tf.expand_dims(tf.reduce_max(self.logit1, axis=1), -1)
            si = tf.nn.softmax(logits=self.logit1 - logit2, axis=1)
        # si = tf.nn.softmax(tf.reshape(tf.matmul(fid, tf.expand_dims(qi, -1)), [-1,self.dLen]), 1)
        return si

    def topm(self, f, g, m=50):
        # entity = tf.concat([f, g], axis=1)
        # product = tf.matmul(entity, tf.transpose(entity, perm=[0, 2, 1]))
        product = tf.matmul(f, tf.transpose(f, perm=[0, 2, 1]))
        a1 = tf.reshape(tf.nn.top_k(product, k=m).values, shape=[-1,1])
        # a2 = tf.reshape(tf.tile(product, multiples=[1,1,m]), shape=[-1, self.qLen + self.dLen])
        a2 = tf.reshape(tf.tile(product, multiples=[1,1,m]), shape=[-1, self.dLen])
        # a3 = tf.reduce_sum(tf.cast(tf.reshape(tf.equal(a1, a2), [-1, m, self.qLen + self.dLen]), tf.float32), axis=1)
        a3 = tf.reduce_sum(tf.cast(tf.reshape(tf.equal(a1, a2), [-1, m, self.dLen]), tf.float32), axis=1)
        # adj = tf.reshape(a3, [-1, self.qLen + self.dLen, self.qLen + self.dLen])
        adj = tf.reshape(a3, [-1, self.dLen, self.dLen])
        return adj

    def GCN(self, adj,x,idx, shape):
        # adj = adj + tf.eye(self.qLen + self.dLen, batch_shape=[BATCH_SIZE])
        d = self.placeholders['d']
        batch_size = tf.shape(d)[0]
        adj = adj + tf.eye(self.dLen, batch_shape=[batch_size])
        row_sum = tf.reduce_sum(adj, 2)
        D_sqrt = tf.pow(row_sum, -0.5)
        D_mat_sqrt = tf.matrix_diag(D_sqrt)
        D_A_D = tf.matmul(tf.transpose(tf.matmul(adj, D_mat_sqrt), [0, 2, 1]), D_mat_sqrt)
        D_A_D = tf.nn.dropout(D_A_D, self.placeholders['k'])
        # print(D_A_D.get_shape()) # (64, 1548, 1548)

        # theta_1 = tf.get_variable(shape=[self.qLen + self.dLen, graph_dim], name='theta_1',
        #                           initializer=tf.glorot_uniform_initializer())
        theta = tf.get_variable(shape=shape, name='theta_{}'.format(idx),
                                  initializer=tf.glorot_uniform_initializer())
        self.vars['GCN_weight_{}'.format(idx)] = theta
        # theta_2 = tf.get_variable(shape=[graph_dim, graph_dim], name='theta_2',
        #                           initializer=tf.glorot_uniform_initializer())
        # former = tf.matmul(adj, theta_1)
        former = tf.matmul(x, tf.tile(tf.expand_dims(theta, 0), multiples=[batch_size, 1, 1]))
        ge = tf.nn.relu(tf.matmul(D_A_D, former))
        return ge

    def _loss(self):
        a = self.placeholders['a']
        d = self.placeholders['d']
        loss = 0
        for i in self.vars.values():
            loss += weight_decay * tf.nn.l2_loss(i)
        loss += tf.reduce_mean(-tf.log(tf.reduce_sum(tf.cast(tf.equal(tf.expand_dims(a, -1), d), tf.float32) * self.si, 1) + tf.constant(1e-12)))  # its shape:[batch_size,1]
        return loss

    def GradientClip(self):
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        vars = [x[1] for x in grads_and_vars]
        gradients = [x[0] for x in grads_and_vars]
        self.clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=MAX_NORM)
        self.grad_norm = tf.global_norm(self.clipped)
        opt = self.optimizer.apply_gradients(zip(self.clipped, vars))
        return opt

    def lrDecay(self,_lr):
        self.global_step = tf.Variable(0,trainable=False)
        lr = tf.train.exponential_decay(_lr, self.global_step, 650,0.98,staircase=True)
        return lr
    def build(self):
        self.placeholders = self.BuildP()
        self.si = self.predict()
        self.loss = self._loss()
        tf.summary.scalar('loss', self.loss)
        self.lr = self.lrDecay(self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt = self.optimizer.minimize(self.loss, global_step=self.global_step)


# if __name__ == '__main__':
    # model = Reader(embedding_dim=384, hidden_dim=384, dLen=1338, qLen=210, learning_rate=learning_rate)
    # model.build()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # should release commanded sentence above
    #     print(sess.run(model.embeddings))
