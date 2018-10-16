

class MCTS:
    def __init__(
        self,
        original_image,
        global_feature,
        crop_img,
        sess_policy,
        sess_evaluate,
        tau,
        c_puct,
        num_virtual_threads=0,
        virtual_loss=0
    ):
        self.original_image = original_image
        self.global_feature = global_feature

        self.root = Node(
            crop_img,
            hidden_state=np.zeros([1, 1024]),
            cell_state=np.zeros([1, 1024]),
            ratio=np.repeat([[0, 0, 20, 20]], 1, axis=0),
            terminal=False
        )
        self.sess = {'policy': sess_policy, 'evaluate': sess_evaluate}
        self.param = {
            'tau': tau,
            'c_puct': c_puct,
            'num_virtual_threads': num_virtual_threads,
            'virtual_loss': virtual_loss
        }
        with self.sess['evaluate'].as_default():
            self.baseline = self.sess['evaluate'].run(
                score_func,
                feed_dict={
                    image_placeholder_evaluate: original_img
                }
            )[0]

    def _get_evaluation(self, crop_image):
        with self.sess['evaluate'].as_default():
            return self.sess['evaluate'].run(
                score_func, feed_dict={
                    image_placeholder_evaluate: crop_img
                }
            )[0] - self.baseline

    def _get_action_prob(self, crop_image, hidden_state, cell_state):
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
            if leaf.state['terminal'] is False:
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
            #'the rollout statistics are updated as if it has lost n_vl games'
            if s.parent_info is not None:
                s.parent_info['statistics']['Nsa'] += self.param['virtual_loss']
                s.parent_info['statistics']['Wsa'] -= self.param['virtual_loss']
                s.parent_info['statistics']['Qsa'] = s.parent_info[
                    'statistics'
                ]['Wsa'] / s.parent_info['statistics']['Nsa']
            s = max_child
        return s

    def _rollout(self, leaf):
        crop_img, hidden_state, cell_state, ratio, terminal = (
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state'], leaf.state['ratio'],
            leaf.state['terminal']
        )
        while terminal is False:
            #sample an action
            action_prob, hidden_state, cell_state = self._get_action_prob(
                crop_img, hidden_state, cell_state
            )
            action_prob = np.reshape(action_prob, num_actions)

            action_ls = np.random.choice(num_actions, 1, action_prob)
            ratio_ls = np.repeat(ratio, 1, axis=0)
            terminal_ls = np.zeros(1, dtype=bool)
            ratio_ls, terminal_ls = command2action(
                action_ls, ratio_ls, terminal_ls
            )
            bbox = generate_bbox(self.original_image, ratio_ls)

            crop_img_ls = [
                crop_input(self.original_image, bbox)
                if terminals_ls[0] is False else crop_img
            ]

            crop_img = crop_img_ls[0]
            ratio = ratio_ls[0]
            terminal = terminal_ls[0]

        #reach a terminal state, then compute ranking scores
        return self._get_evaluation(crop_img)

    def _expand(self, leaf):
        if leaf.state['terminal'] is True:
            return
        action_prob, hidden_state, cell_state = self._get_action_prob(
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state']
        )

        action_ls = np.array(range(num_actions))
        ratio_ls = np.repeat(
            leaf.state['ratio'], num_actions, axis=0
        )  #intial ratio is of batch_size 1
        terminal_ls = np.zeros(num_actions, dtype=bool)
        ratio_ls, terminal_ls = command2action(action_ls, ratio_ls, terminal_ls)
        bbox = generate_bbox(self.original_image, ratio_ls)

        crop_img_ls = [
            crop_input(self.original_image, bbox)
            if terminal_ls[i] is False else leaf.state['crop_img']
            for i in range(num_actions)
        ]
        s.children = []
        for i in range(num_actions):
            s.children.append(
                Node(
                    crop_img_ls[i],
                    hidden_state=hidden_state,
                    cell_state=cell_state,
                    ratio=ratio_ls[i],
                    terminal=terminal_ls[i],
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
        if self.root.state is not None and self.root.state['terminal'] is True:
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
        
        
