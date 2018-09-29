import tensorflow as tf
import sys
import os
import shutil
import pandas as pd

# change this as you see fit
filepath = sys.argv[1]
print(filepath)
# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
resdata={}
with tf.Session() as sess:
    pathDir = os.listdir(filepath)
    for imgfile in pathDir:
        print(imgfile)
        try:
            scorecount = 0
            image_path = os.path.join('%s/%s' % (filepath, imgfile))
            print(image_path)
            #if os.path.splitext(image_path)[1] == ".bmp" or os.path.splitext(image_path)[1] == ".BMP" or os.    path.splitext(image_path)[1] == ".gif" or os.path.splitext(image_path)[1] == ".GIF" :
            #    new_path=os.path.splitext(image_path)[0]+".jpg"
            if os.path.splitext(image_path)[1] == ".jpg":
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
                
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                
                predictions = sess.run(softmax_tensor, \
                         {'DecodeJpeg/contents:0': image_data})
                
                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                
                # submitfile = open(".\submit.csv","a",encoding="utf-8")
                # submitfile.write("img_path,tags\n")

                # submitfile.write("\n"+os.path.basename(image_path))
                filename = os.path.basename(image_path)
                resdata[filename]=""
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    scorecount += 1
                    
                    if scorecount > 80:
                        continue
                    if scorecount == 1:
                        resdata[filename] = human_string
                    else:
                        resdata[filename] = resdata[filename]+","+human_string
                

                filename = "results.txt"    
                with open(filename, 'a+') as f:
                    f.write('\n**%s**\n' % (image_path))
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        f.write('%s (score = %.5f)\n' % (human_string, score))
        
        except Exception as e:
            print(image_path)
            print(e)

a = []
b = []
for res in resdata:
    #print(res,resdata[res])
    a.append(res)
    b.append(resdata[res])
dataframe = pd.DataFrame({'img_path':a,'tags':b})
dataframe.to_csv("submit.csv",index=False,sep=',')
    

