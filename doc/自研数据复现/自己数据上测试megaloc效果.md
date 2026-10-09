
# 20290928

现在,我想在自己采集的数据集上测试Megaloc的效果。

数据录的包我存储在/data/data/reLocationData/zc_data，相关传感器的参数见/data/data/location_data/zc_data/研发大楼/022419/calib/hesai_022419。用20260921_064638.mcap用来建图，其余两个包用于查询，给出模型最终重定位的精度。

另一个场景的包的路径在/data/data/reLocationData/zc_data/研发大楼数据这个数据中进行测试reloc3R的效果，将20260923_055425_real_robot_record_external_lz4来建图，20260923_060415_real_robot_record_external_lz4作为查询

环境使用/data/software/conda/anaconda3/envs/mv2d

先分析应该怎么做，然后进行测试，给出最终的测试结果。将结果放在doc/自研数据复现中，用文档保存下来，文档的结尾要加上时间戳。

对于实现可能遇到的一些问题，你可以参考/data/code/reLocation-all/reloc3r/docs/zc数据集上测试/zc数据集上测试reloc3r效果_qwen3.8-flash-202609281505.md这个文档，它是在这些数据集上测试另一个模型得到的记录。

给出方案后再实现这个方案。
