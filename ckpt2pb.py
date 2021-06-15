import tensorflow as tf
import re
from tensorflow.python.framework import graph_util

def main():
	ckpt_path = 'policies/walking/alternating_legs/pupper-mix-5m/model.ckpt-10000000'
	out_path = 'export/0'
	# tf.reset_default_graph()
	# loaded_graph = tf.Graph()
	
	with tf.Session() as sess:
		# Restore variables from disk.
		saver = tf.train.import_meta_graph(ckpt_path + '.meta')
		saver.restore(sess, ckpt_path)
		print("Model restored.")
		# print(tf.get_collection('variables'))
		print(tf.get_default_graph().get_all_collection_keys())
		output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
		print(output_node_names)
		frozen_graph_def = tf.graph_util.convert_variables_to_constants(
	    sess,
	    sess.graph_def,
	    output_node_names)

		# Save the frozen graph
		with open('output_graph.pb', 'wb') as f:
		  f.write(frozen_graph_def.SerializeToString())
		# builder = tf.saved_model.builder.SavedModelBuilder(out_path)
		# builder.add_meta_graph_and_variables(sess,
		#								[tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING],
		#								strip_default_attrs=True)
		# builder.save()

if __name__ == '__main__':
	main()