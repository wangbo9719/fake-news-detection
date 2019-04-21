import pickle

if 1:
	f = open('article_index_subject_dict', 'rb')
	d = pickle.load(f)
	f.close()

	index = 1000000000
	subject_dict = {}
	f = open('edge', 'w')
	for article in d:
		for subject in d[article]:
			if subject not in subject_dict:
				subject_dict[subject] = index
				index += 1
			f.write(str(article) + '\t' + str(subject_dict[subject]) + '\t' + str(1) + '\n')
			f.flush()
	f.close()

	print(subject_dict)

	f = open('subject_index', 'w')
	for subject in subject_dict:
		f.write(str(subject) + '\t' + str(subject_dict[subject]) + '\n')
	f.close()
