

class MCTS:
    def __init__(
        self,
        original_image,
        sess_policy,
        sess_evaluate,
        c_puct,
        num_virtual_threads=0,
        virtual_loss=0
    ):
        original_image = np.expand_dims(original_image, axis=0)
        self.original_image = original_image
        with sess_policy.as_default():
            self.global_feature = sess_policy.run(
                global_feature_placeholder,
                feed_dict={
                    image_placeholder_policy: original_image
                }
            )

        self.root = Node(
            crop_img=original_image,
            hidden_state=np.zeros([1, 1024]),
            cell_state=np.zeros([1, 1024]),
            ratio=np.repeat([[0, 0, 20, 20]], 1, axis=0),
            terminal=np.zeros(1, dtype=bool)
        )

        self.param = {
            'c_puct': c_puct,
            'num_virtual_threads': num_virtual_threads,
            'virtual_loss': virtual_loss
        }
        with sess_evaluate.as_default():
            self.baseline = self.sess['evaluate'].run(
                score_func,
                feed_dict={
                    image_placeholder_evaluate: original_image
                }
            )[0]
        self.sess = {'policy': sess_policy, 'evaluate': sess_evaluate}

    def _get_evaluation(self, crop_img):
        with self.sess['evaluate'].as_default():
            return self.sess['evaluate'].run(
                score_func, feed_dict={
                    image_placeholder_evaluate: crop_img
                }
            )[0] - self.baseline

    def _get_action_prob(self, crop_img, hidden_state, cell_state):
        with self.sess['policy'].as_default():
            return self.sess['policy'].run(
                (action, h, c),
                feed_dict={
                    image_placeholder_policy: crop_img,
                    global_feature_placeholder: self.global_feature,
                    h_placeholder: hidden_state,
                    c_placeholder: cell_state
                }
            )

    def simulate_once(self):
        leaf_ls = []
        for _ in range(self.param['num_virtual_threads']):
            leaf_ls.append(self._select())
        visit_times = collections.Counter(leaf_ls)

        value_ls = []
        for leaf in visit_times:
            value_ls.append(self._rollout(leaf))
            self._expand(leaf)
            if not leaf.state['terminal'][0]:
                leaf.state = None  #leaf is updated to an internal node, if it is not of terminal state
        for i, (leaf, freq) in enumerate(visit_times.items()):
            self._backup(leaf, freq, value_ls[i])

    def _select(self):
        s = self.root
        while s.state is None:
            visit_total = 0
            for c in s.children:
                visit_total += c.parent_info['statistics']['Nsa']
            max_child = None
            for c in s.children:
                if max_child is None or (
                    max_child.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * max_child.parent_info['statistics']
                    ['Psa'] * math.sqrt(visit_total) /
                    (1 + max_child.parent_info['statistics']['Nsa'])
                ) < (
                    c.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * c.parent_info['statistics']['Psa'] *
                    math.sqrt(visit_total) /
                    (1 + c.parent_info['statistics']['Nsa'])
                ):
                    max_child = c
            s = max_child
            #'the rollout statistics are updated as if it has lost n_vl games'
            s.parent_info['statistics']['Nsa'] += self.param['virtual_loss']
            s.parent_info['statistics']['Wsa'] -= self.param['virtual_loss']
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
        return s

    def _rollout(self, leaf):
        #notice the np.copy
        crop_img, hidden_state, cell_state, ratio, terminal = (
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state'], np.copy(leaf.state['ratio']),
            np.copy(leaf.state['terminal'])
        )

        while not terminal[0]:
            #sample an action
            action_prob, hidden_state, cell_state = self._get_action_prob(
                crop_img, hidden_state, cell_state
            )

            action_sample = np.random.choice(
                num_actions, 1, p=np.reshape(action_prob, (-1))
            )
            ratio, terminal = command2action(action_sample, ratio, terminal)
            if not terminal[0]:
                bbox = generate_bbox(self.original_image, ratio)
                crop_img = crop_input(self.original_image, bbox)

        #reach a terminal state, then compute ranking scores
        return self._get_evaluation(crop_img)

    def _expand(self, leaf):
        if leaf.state['terminal'][0]:
            return
        action_prob, hidden_state, cell_state = self._get_action_prob(
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state']
        )
        action_prob = np.reshape(action_prob, (-1))

        action_ls = np.array(range(num_actions), dtype=np.int32)
        ratio_ls = np.repeat(
            leaf.state['ratio'], num_actions, axis=0
        )  #intial ratio is of batch_size 1
        terminal_ls = np.repeat(
            leaf.state['terminal'], num_actions, axis=0
        )  #intial terminal is of batch_size 1
        ratio_ls, terminal_ls = command2action(action_ls, ratio_ls, terminal_ls)
        bbox_ls = generate_bbox(
            np.repeat(self.original_image, num_action, axis=0), ratio_ls
        )
        crop_img_ls = crop_input(self.original_image, bbox_ls)

        leaf.children = []
        for i in range(num_actions):
            s.children.append(
                Node(
                    crop_img=np.expand_dims(crop_img_ls[i], axis=0),
                    hidden_state=hidden_state,
                    cell_state=cell_state,
                    ratio=np.expand_dims(ratio_ls[i], axis=0),
                    terminal=np.expand_dims(terminal_ls[i], axis=0),
                    parent=s,
                    Nsa=0,
                    Wsa=0,
                    Qsa=0,
                    Psa=action_prob[i]
                )
            )

    def _backup(self, leaf, freq, value):
        s = leaf
        while s.parent_info is not None:
            s.parent_info['statistics'
                         ]['Nsa'] += (-self.param['virtual_loss'] + 1) * freq
            s.parent_info['statistics'
                         ]['Wsa'] += (self.param['virtual_loss'] + value) * freq
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
            s = s.parent_info['parent']

    def get_crop_box(self):
        if self.root.state is None:  #require a leaf node
            return None
        return generate_bbox(self.original_image, self.root.state['ratio'])

    def reach_terminal(self):
        if self.root.state is not None and self.root.state['terminal'][0
                                                                      ] is True:
            return True
        return False

    def act_one_step(self):
        #select the action with maximum visit count
        if self.root.state is not None:  #a leaf node
            return -1

        #root is an internal node
        max_child = None
        action = -1
        for i, c in enumerate(self.root.children):
            if max_child is None or max_child.parent_info['statistics'][
                'Nsa'
            ] < c.parent_info['statistics']['Nsa']:
                max_child = c
                action = i
        self.root = max_child
        root.parent_info = None
        return action
