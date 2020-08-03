## Update 03 August 2020
A complete project (entire code and description) of the improvement of this paper will be published next year!

## Note
According to the requests from some researchers who want to re-implement the work **Anomaly Detection in Video Sequence with Appearance-Motion Correspondence (ICCV2019)** ([official page](http://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.html) | [arXiv](https://arxiv.org/pdf/1908.06351.pdf) | [demo](https://youtu.be/PaUenXHHzuw)) in other deep learning frameworks, I tried to isolate the related piece of code from the whole (messy) project.

The network architectures (generator and discriminator) were implemented with Tensorflow and can be found in [GAN_tf.py](./GAN_tf.py). Some related functions are also provided to give the big picture of the work (but they are not encouraged to use because some of them were discarded during the work).

Finally, sorry for such non-optimized code since I do not have time to improve it for a final version.

```
@InProceedings{Nguyen_2019_ICCV,
  author = {Nguyen, Trong-Nguyen and Meunier, Jean},
  title = {Anomaly Detection in Video Sequence With Appearance-Motion Correspondence},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```
