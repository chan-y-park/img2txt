import random
import string
import json

import numpy as np
import nltk

from PIL import Image


class ImageCaptionDataset:
    def __init__(
        self,
        image_dir=None,
    ):
        # Captions of a given image, its file name is the key of this dict.
        self._captions = {}
        # A list of (image_file_name, caption_id)
        self._data = []
        self._image_dir = image_dir
        self.name = None

    def get_image_ids(self):
        return list(self._captions.keys())

    def get_size(self):
        return len(self._data)

    def get_captions(self, img_id):
        return self._captions[img_id]

    def get_random_image(self):
        img_id, _ = random.choice(self._data)
        return {
            'id': img_id,
            'image': self.get_image(img_id),
        }

    def get_image(self, img_id):
        return Image.open(self._image_dir + img_id)

    def get_example(self, img_id=None):
        if img_id is None:
            rd = self.get_random_image()
            img_id = rd['id']
            img = rd['image']
        else:
            img = self.get_image(img_id)
        return {
            'image': img, 
            'captions': self.get_captions(img_id),
        }


class PASCAL(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
    ):
        super().__init__(image_dir)
        self.name = 'pascal'

        img_path_start_string = '<td><img src="pascal-sentences_files/'
        img_path_end_string = '"></td>\n'

        caption_start_strings = ['<tbody><tr><td> ', '<tr><td> ']
        caption_end_string = '</td></tr>\n'

        item_end_string = '</tbody></table></td>\n'

        with open(caption_file, 'r') as fp:
            line_number = 0
            for line in fp:
                line_number += 1
                
                if (
                    line[:len(img_path_start_string)] == img_path_start_string
                ):
                    img_path = line[len(img_path_start_string)
                                    :-len(img_path_end_string)]
                    self._captions[img_path] = []

                    while (line != item_end_string):
                        line = fp.readline()
                        line_number += 1
                        for start_string in caption_start_strings:
                            if (line[:len(start_string)] == start_string):
                                line = line[len(start_string):]
                                caption = ''
                                while(line[-len(caption_end_string):]
                                      != caption_end_string):
                                    caption += line.strip()
                                    line = fp.readline()
                                    line_number += 1
                                caption += line[:-len(caption_end_string)]
                                caption = caption.rstrip(
                                    string.whitespace + '.'
                                )
                                self._captions[img_path].append(caption)

                                caption_id = len(self._captions[img_path]) - 1
                                self._data.append(
                                    (img_path, caption_id)
                                )

        # End of caption file preprocessing.


class Flickr(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
        name='flickr',
    ):
        super().__init__(image_dir)
        self.name = name
        prev_img_id = None

        with open(caption_file, 'r') as fp:
            for line in fp:
                filename_length = line.find('#')
                img_id = line[:filename_length]
                if img_id != prev_img_id:
                    self._captions[img_id] = []
                    prev_img_id = img_id

                raw_caption = line[filename_length + 3:]
                caption = raw_caption.rstrip(
                    string.whitespace + '.'
                )
                self._captions[img_id].append(caption)

                caption_id = len(self._captions[img_id]) - 1
                self._data.append(
                    (img_id, caption_id)
                )

    def split_dataset(
        self,
        train_file_path,
        dev_file_path,
        test_file_path,
    ):
        train_dataset = ImageCaptionDataset(image_dir=self._image_dir)
        dev_dataset = ImageCaptionDataset(image_dir=self._image_dir)
        test_dataset = ImageCaptionDataset(image_dir=self._image_dir)
        for file_path, sub_dataset in (
            (train_file_path, train_dataset),
            (dev_file_path, dev_dataset),
            (test_file_path, test_dataset),
        ):
            with open(train_file_path, 'r') as fp:
                for line in fp:
                    img_id = line.rstrip()
                    captions = self._captions[img_id]
                    sub_dataset._captions[img_id] = captions
                    for caption_id in range(len(captions)):
                        sub_dataset._data.append(
                            (img_id, caption_id)
                        )
        train_dataset.name = self.name + '_training'
        dev_dataset.name = self.name + '_validation'
        test_dataset.name = self.name + '_test'
        return train_dataset, dev_dataset, test_dataset


class MSCOCO(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
        name='ms_coco',
    ):
        super().__init__(image_dir)
        self.name = name
        image_file_name_of_id = {}

        with open(caption_file, 'r') as fp:
            captions_json = json.load(fp)

        for image in captions_json['images']:
            image_file_name_of_id[image['id']] = image['file_name']

        annotations = captions_json['annotations']
        for annotation in annotations:
            caption = annotation['caption']
            image_id = image_file_name_of_id[annotation['image_id']]
            try:
                self._captions[image_id].append(caption)
            except KeyError:
                self._captions[image_id] = [caption]
            caption_id = len(self._captions[image_id]) - 1
            self._data.append(
                (image_id, caption_id)
            )


class Vocabulary:
    def __init__(
        self,
        min_word_count=1,
        dataset=None,
        file_path=None,
        name=None,
    ):
        self.name = name
        self._word_count = {}
        self._id_of_word = {}
        self._word_of_id = {}
        self._min_word_count = min_word_count

        # NOTE: Negative word ids don't work
        # with sparse argmax-type functions.
        self.start_word = '<bos>'
        self.start_word_id = 0

        self.end_word = '<eos>'
        self.end_word_id = 1

        self.unknown_word = '<unk>'
        self.unknown_word_id = 2

        self._id_of_word[self.start_word] = self.start_word_id
        self._id_of_word[self.end_word] = self.end_word_id
        self._id_of_word[self.unknown_word] = self.unknown_word_id

        self._word_of_id[self.start_word_id] = self.start_word
        self._word_of_id[self.end_word_id] = self.end_word
        self._word_of_id[self.unknown_word_id] = self.unknown_word

        if file_path is not None:
            with open(file_path, 'r') as fp:
                saved_vocabulary = json.load(fp)
            self._word_count = saved_vocabulary['word_count']
            self._id_of_word = saved_vocabulary['id_of_word']
            for word_id, word_str in saved_vocabulary['word_of_id'].items():
                self._word_of_id[int(word_id)] = word_str
            self._min_word_count = saved_vocabulary['min_word_count']
        else:
            self._count(dataset)
            self._build()

    def save(self, file_path):
        serialized_data = {
            'word_count': self._word_count,
            'id_of_word': self._id_of_word,
            'word_of_id': self._word_of_id,
            'min_word_count': self._min_word_count,
        }
        with open(file_path, 'w') as fp:
            json.dump(serialized_data, fp)


    def get_id_of_word(self, word):
        try:
            word_id = self._id_of_word[word]
        except KeyError:
            word_id = self.unknown_word_id
        return word_id

    def get_word_of_id(self, word_id):
        return self._word_of_id[word_id]

    def get_size(self):
        return len(self._id_of_word)

    def get_word_count(self, word):
        return self._word_count[word]

    def get_sorted_words(self):
        return sorted(
            self._word_count.keys(),
            key=self.get_word_count,
            reverse=True,
        )

    def tokenize(self, sentence):
        tokens = (
            [self.start_word]
            + [w.lower() for w in nltk.word_tokenize(sentence)]
            + [self.end_word]
        )
        return tokens

    def _count(self, dataset):
        for img_id, caption_id in dataset._data:
            caption = dataset._captions[img_id][caption_id]
            tokens = self.tokenize(caption)
            for word in tokens:
                try:
                    self._word_count[word] += 1
                except KeyError:
                    self._word_count[word] = 1

    def _build(self):
        word_id = 3
        self._word_count[self.unknown_word] = 0
        # Start with <bos>, <eos>, and <unk>.
        for word, count in self._word_count.items():
            if (
                word == self.start_word
                or word == self.end_word
            ):
                pass
            elif count < self._min_word_count:
                self._word_count[self.unknown_word] += 1
            else:
                self._id_of_word[word] = word_id
                self._word_of_id[word_id] = word
                word_id += 1

    def get_preprocessed_sentence(self, raw_sentence):
        return [self.get_id_of_word(word)
                for word in self.tokenize(raw_sentence)]

    def get_sentence_from_word_ids(self, word_ids):
#        words = []
#        for word_id in word_ids:
#            if word_id == self.end_word_id:
#                break
#            else:
#                word = self.get_word_of_id(word_id)
#                words.append(word)
        words = [
            self.get_word_of_id(word_id)
            for word_id in word_ids
        ]
            
        sentence = ' '.join(words)
        return sentence

    def get_postprocessed_sequence(self, sequence):
        post_seq = []
        for word_id in sequence:
            if word_id == self.end_word_id:
                break
            else:
                post_seq.append(word_id)
        return post_seq


def evaluate_bleu_scores(dataset):
    pass
