import pandas as pd

query_list =  pd.read_csv("data/query.csv", header = None).values # Object list to sample from 


def filter_objects_v2(instance_segmentation, visibleAreaThreshold=0.01):
	'''
	args:
		@metaDataObjects of type event.instance_detections2D
		values of format [start_x, start_y, end_x, end_y].
		@visibleAreaThreshold of type float. Any object larger than this area is returned.
	returns:
		list of dicts of objects that are larger than the required area. Same as event.metadata['objects']
	'''
	total_area = 300 * 300
	visibleObjects = []
	notVisibleMetaDataObjectsList = [["ObjectType", "areaFraction"]]
	for key, bb in instance_segmentation.items():
		if key.split('|')[0] not in query_list:
			continue
		area = abs(bb[2] - bb[0]) * abs(bb[3] - bb[1])
		if area/total_area > visibleAreaThreshold:
			visibleObjects.append(key)
		else:
			notVisibleMetaDataObjectsList.append([key, area/total_area])
	# print(f"Not Visible List filtered out for threshold {visibleAreaThreshold}", notVisibleMetaDataObjectsList)
	return visibleObjects