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
        #no resize is useful for generate_bbox and crop_input
        self.original_image = np.expand_dims(
            original_image, axis=0
        )  #no resize!!!
        self.dummy_original_image_ls = [original_image] * num_virtual_threads

        #the neural networks require the resized image!!!
        original_image = transform.resize(
            original_image, (227, 227), mode='constant'
        )  #after resize
        original_image = np.expand_dims(original_image, axis=0)

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
            self.baseline = sess_evaluate.run(
                score_func,
                feed_dict={
                    image_placeholder_evaluate: original_image
                }
            )[0]
        #debug
        print('baseline {}'.format(self.baseline))
        self.sess = {'policy': sess_policy, 'evaluate': sess_evaluate}

    def _get_evaluation(self, crop_img):
        with self.sess['evaluate'].as_default():
            values = self.sess['evaluate'].run(
                score_func, feed_dict={
                    image_placeholder_evaluate: crop_img
                }
            )
        for i in range(len(values)):
            values[i] -= self.baseline
        return values

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

    def simulate_many_times(self, num_times):
        assert num_times > 1

        for _ in range(num_times):
            self._simulate_once()

    def _simulate_once(self):
        leaf_ls = []
        for _ in range(self.param['num_virtual_threads']):
            leaf_ls.append(self._select())

        value_ls = self._rollout_ls(leaf_ls)

        for i, leaf in enumerate(leaf_ls):
            self._backup(leaf, value_ls[i])

        leaf_set = set(leaf_ls)
        self._expand_ls(leaf_set)
        for leaf in leaf_set:
            if not leaf.state['terminal'][0]:
                leaf.state = None

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

    def _rollout_ls(self, leaf_ls):
        #initialize
        crop_img_ls, hidden_state_ls, cell_state_ls, ratio_ls, terminal_ls = [
            [None] * len(leaf_ls)
        ] * 5

        #notice the np.copy
        for i, leaf in enumerate(leaf_ls):
            crop_img_ls[i], hidden_state_ls[i], cell_state_ls[i], ratio_ls[
                i
            ], terminal_ls[i] = (
                leaf.state['crop_img'][0], leaf.state['hidden_state'][0],
                leaf.state['cell_state'][0], np.copy(leaf.state['ratio'][0]),
                leaf.state['terminal'][0]
            )
        index_ls = [_ for _ in range(len(leaf_ls))]

        final_img_ls = [None] * len(leaf_ls)

        while len(crop_img_ls) > 0:
            #purge
            purge_img_ls, purge_hidden_ls, purge_cell_ls, purge_ratio_ls, purge_terminal_ls, purge_index = [
                list()
            ] * 6

            for i, terminal in enumerate(terminal_ls):
                if not terminal:
                    purge_img_ls.append(crop_img_ls[i])
                    purge_hidden_ls.append(hidden_state_ls[i])
                    purge_cell_ls.append(cell_state_ls[i])
                    purge_ratio_ls.append(ratio_ls[i])
                    purge_terminal_ls.append(terminal_ls[i])

                    purge_index.append(index_ls[i])
                else:
                    final_img_ls[index_ls[i]] = crop_img_ls[i]

            if len(purge_img_ls) == 0:
                break
            #compute action distributions
            action_ls = [None] * len(purge_img_ls)
            for i in range(len(purge_img_ls) // BATCH_SIZE + 1):
                start, end = (
                    i * BATCH_SIZE,
                    min((i + 1) * BATCH_SIZE, len(purge_img_ls))
                )
                action_prob_ls, purge_hidden_ls[start:end], purge_cell_ls[
                    start:end
                ] = self._get_action_prob(
                    purge_img_ls[start:end], purge_hidden_ls[start:end],
                    purge_cell_ls[start:end]
                )

                action_ls[start:end] = [
                    np.random.choice(num_actions, 1, p=action_prob_ls[k])[0]
                    for k in range(len(action_prob_ls))
                ]

            ratio_ls, terminal_ls = command2action(
                action_ls, purge_ratio_ls, purge_terminal_ls
            )

            bbox = generate_bbox(
                self.dummy_original_image_ls[:len(purge_img_ls)], ratio_ls
            )
            crop_img_ls = crop_input(
                self.dummy_original_image_ls[:len(purge_img_ls)], bbox
            )

            hidden_state_ls, cell_state_ls = (purge_hidden_ls, purge_cell_ls)
            index_ls = purge_index

        value_ls = [None] * len(leaf_ls)
        for i in range(len(leaf_ls) // BATCH_SIZE + 1):
            start, end = (
                i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(leaf_ls))
            )
            value_ls[start:end] = self._get_evaluations(final_img_ls[start:end])
        return value_ls

    def _expand_ls(self, leaf_ls):
        purge_img_ls, purge_hidden_ls, purge_cell_ls, purge_leaf_ls = [list()
                                                                      ] * 4

        for leaf in leaf_ls:
            if not leaf.state['terminal'][0]:
                purge_leaf_ls.append(leaf)

                purge_img_ls.append(leaf.state['crop_img'][0])
                purge_hidden_ls.append(leaf.state['hidden_state'][0])
                purge_cell_ls.append(leaf.state['cell_state'][0])
        if len(purge_leaf_ls) == 0:
            return

        action_prob_ls = [None] * len(purge_leaf_ls)
        for i in range(len(purge_img_ls) // BATCH_SIZE + 1):
            start, end = (
                i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(purge_img_ls))
            )
            action_prob_ls[start:end], purge_hidden_ls[
                start:end
            ], purge_cell_ls[start:end] = self._get_action_prob(
                purge_img_ls[start:end], purge_hidden_ls[start:end],
                purge_cell_ls[start:end]
            )

        dummy_action_ls = [_ for _ in range(num_actions)]
        dummy_terminal_ls = [False] * num_actions

        for i, leaf in enumerate(purge_leaf_ls):
            ratio_leaf = np.repeat(leaf.state['ratio'], num_actions, axis=0)
            ratio_leaf, dummy_terminal_ls = command2action(
                dummy_action_ls, ratio_leaf, dummy_terminal_ls
            )
            box_ls = generate_bbox(
                self.dummy_original_image_ls[:num_actions], ratio_leaf
            )
            crop_img_ls = crop_input(
                self.dummy_original_image_ls[:num_actions], box_ls
            )

            leaf.children = list()
            for k in range(num_actions):
                leaf.children.append(
                    Node(
                        crop_img=list(crop_img_ls[k]),
                        hidden_state=list(purge_hidden_ls[i]),
                        cell_state=list(purge_cell_ls[i]),
                        ratio=list(ratio_leaf[k]),
                        terminal=list(dummy_terminal_ls[k]),
                        parent=leaf,
                        Nsa=0,
                        Wsa=0,
                        Qsa=0,
                        Psa=action_prob_ls[i][k]
                    )
                )
            dummy_terminal_ls[-1] = False

    def _backup(self, leaf, value):
        s = leaf
        while s.parent_info is not None:
            s.parent_info['statistics']['Nsa'
                                       ] += (-self.param['virtual_loss'] + 1)
            s.parent_info['statistics'
                         ]['Wsa'] += (self.param['virtual_loss'] + value)
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
            s = s.parent_info['parent']

    def get_crop_box(self):
        if self.root.state is None:  #require a leaf node
            return None
        return generate_bbox(self.original_image, self.root.state['ratio'])[0]

    def reach_terminal(self):
        if self.root.state is not None and self.root.state['terminal'][0]:
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
                #             if max_child is None or max_child.parent_info['statistics'][
                #                 'Qsa'
                #             ] < c.parent_info['statistics']['Qsa']:
                max_child = c
                action = i
        self.root = max_child
        self.root.parent_info = None
        return action

    def debug_print(self):
        if self.root.state is not None:
            print('reach a leaf')
        else:
            for i, c in enumerate(self.root.children):
                print(
                    'Nsa {}, Wsa {}, Qsa {}, Psa {}'.format(
                        c.parent_info['statistics']['Nsa'],
                        c.parent_info['statistics']['Wsa'],
                        c.parent_info['statistics']['Qsa'],
                        c.parent_info['statistics']['Psa']
                    )
                )
                print('above info for child {}'.format(i))
            print('end of the printing this time ...')
