import random
import string
import json

import numpy as np

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
        self._vocabulary = None 

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

    def tokenize(self, sentence):
        # TODO: Use nltk.
        words = sentence.split()
        return (
            [self.start_word]
            + [word.lower().strip(',"<>') for word in words]
            + [self.end_word]
        )

    def get_vocabulary(self, min_word_count=1):
        word_count = {}
        for img_id, captions in self._captions.items():
            for caption in captions:
                words = self.tokenize(caption)
                for word in words:
                    try:
                        word_count[word] += 1
                    except KeyError:
                        word_count[word] = 1

        vocabulary = {
            'id_of_word': {},
            'word_of_id': {},
        }
        # TODO: Set word id in descending order of word count.
        word_id = 0
        for word, count in word_count.items():
            if count >= min_word_count:
                vocabulary['id_of_word'][word] = word_id
                vocabulary['word_of_id'][word_id] = word
                word_id += 1
                        
        return vocabulary, word_count

    def get_vocabulary_size(self):
        if self._vocabulary is None:
            self._vocabulary, _ = self.get_vocabulary()
        return len(self._vocabulary['id_of_word'])

    def get_preprocessed_caption(self, img_id, caption_id):
        if self._vocabulary is None:
            self._vocabulary, _ = self.get_vocabulary()

        raw_caption = self.get_captions(img_id)[caption_id]
        return [self._vocabulary['id_of_word'][word]
                for word in self.tokenize(raw_caption)]

    def get_sentence_from_word_ids(self, word_ids):
        if self._vocabulary is None:
            self._vocabulary, _ = self.get_vocabulary()
        words = []
        for word_id in word_ids:
            if word_id == self._vocabulary['id_of_word'][self.end_word]:
                break
            else:
                word = self._vocabulary['word_of_id'][word_id]
                words.append(word)
            
        sentence = ' '.join(words) + '.'
        return sentence


class Vocabulary:
    def __init__(
        self,
        min_word_count=1,
    ):
        self._id_of_word = {}
        self._word_of_id = {}
        self._size = None
        # XXX: Use special word ids for the following words?
        self.start_word = '<bos>'
        self.end_word = '<eos>'
        self.unknown_word = '<unk>'

    def get_id_of_word(self, word):
        return self._id_of_word(word)

    def get_word_of_id(self, word_id):
        return self._word_of_id(word_id)

    def get_size(self):
        return self._size


class PASCAL(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
    ):
        super().__init__(image_dir)

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
#        self._vocabulary, _ = self.get_vocabulary()


class Flickr(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
    ):
        super().__init__(image_dir)
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
            

class MSCOCO(ImageCaptionDataset):
    def __init__(
        self,
        caption_file=None,
        image_dir=None,
    ):
        super().__init__(image_dir)
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


