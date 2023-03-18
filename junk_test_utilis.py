# table_name = 'prioritised_table'
        # replay_buffer_signature = tensor_spec.from_spec(
        #     self.collect_data_spec)
        # replay_buffer_signature = tensor_spec.add_outer_dims_nest(
        #     replay_buffer_signature,
        #     [])
        # table = reverb.Table(
        #     table_name,
        #     max_size=self.replay_buffer_capacity,
        #     sampler=reverb.selectors.Uiform(),
        #     remover=reverb.selectors.Fifo(),
        #     rate_limiter=reverb.rate_limiters.MinSize(1),
        #     signature=replay_buffer_signature)
        # checkpointer = reverb.checkpointers.DefaultCheckpointer(path=self.train_dir_base)
        #
        # reverb_server = reverb.Server([table], checkpointer=checkpointer)
        #
        # replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        #     self.collect_data_spec,
        #     table_name=table_name,
        #     sequence_length=2,
        #     local_server=reverb_server)
        #
        #
        #
        # rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        #     replay_buffer.py_client,
        #     table_name,
        #     sequence_length=2,
        # )
        #
        # dataset = replay_buffer.as_dataset(
        #     num_parallel_calls=3,
        #     sample_batch_size=self.batch_size,
        #     num_steps=2).prefetch(3)
        #
        # iterator = iter(dataset)

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dims_nest(
            replay_buffer_signature,
            [])

        table = reverb.Table(
            table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)
