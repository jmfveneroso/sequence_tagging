def quick_train(filename, doc, epochs=150):
  create_model()

  with tf.Session() as sess:  
    saver = tf.train.Saver()
    sess.run([tf.initializers.global_variables(), tf.tables_initializer()])

    for step in range(epochs):
      start_time = time.time()
      features, labels = DL().get_doc(filename, doc)
      r, _ = run_step(sess, features, labels, train=True)

      tags = labels
      preds = r[2].tolist()

      m = evaluate(preds, tags, [], verbose=False)
      print_stats(r[0], r[1], time.time() - start_time, step, f1=m['f1'])

      if m['f1'] > 0.95:
        break
    save_path = saver.save(sess, "./checkpoints/model2.ckpt")
    print("Model saved in path: %s" % save_path)

