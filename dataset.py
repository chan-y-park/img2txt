import random
import string
import numpy as np

from PIL import Image


class ImageCaptionDataset:
    def __init__(
        self,
        data_dir=None,
    ):
        self._captions = {}
        self._img_ids = []
        self._data_dir = data_dir
        self._vocabulary = None 
        # XXX: Use special word ids for the following words?
        self.start_word = '<bos>'
        self.end_word = '<eos>'
        self.unknown_word = '<unk>'

    def get_captions(self, img_id):
        return self._captions[img_id]

    def get_random_image(self, img_id=None):
        if img_id is None:
            img_id = random.choice(self._img_ids)
        return {
            'id': img_id,
            'image': self.get_image(img_id),
        }

    def get_image(self, img_id, to_array=False, size=None):
        image = Image.open(self._data_dir + img_id)
        if size is not None:
            image = image.resize((size, size))
        if to_array:
            image = (np.array(image, dtype=np.float32)
                     / np.iinfo(np.uint8).max)
        return image

    def get_example(self, img_id=None):
        rd = self.get_image(img_id)
        return {
            'image': rd['image'], 
            'captions': self.get_captions(rd['id']),
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
#                    try:
#                        frequency[word_id] += 1
#                    except KeyError:
#                        vocabulary['id_of_word'][word] = word_id
#                        vocabulary['word_of_id'][word_id] = word
#                        frequency[word_id] = 1
#                        word_id += 1
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
        words = []
        for word_id in word_ids:
            if word_id == self._vocabulary['id_of_word'][self.end_word]:
                break
            else:
                word = self._vocabulary['word_of_id'][word_id]
                words.append(word)
            
        sentence = ' '.join(words) + '.'
        return sentence


class PASCAL(ImageCaptionDataset):
    def __init__(
        self,
        caption_filename='pascal-sentences.html',
        data_dir='./',
    ):
        super().__init__(data_dir)

        img_path_start_string = '<td><img src="'
        img_path_end_string = '"></td>\n'

        caption_start_strings = ['<tbody><tr><td> ', '<tr><td> ']
        caption_end_string = '</td></tr>\n'

        item_end_string = '</tbody></table></td>\n'

        with open(data_dir + caption_filename) as f:
            line_number = 0
            for line in f:
                line_number += 1
                
                if (
                    line[:len(img_path_start_string)] == img_path_start_string
                ):
                    img_path = line[len(img_path_start_string)
                                    :-len(img_path_end_string)]
                    self._captions[img_path] = []
                    self._img_ids.append(img_path)

                    while (line != item_end_string):
                        line = f.readline()
                        line_number += 1
                        for start_string in caption_start_strings:
                            if (line[:len(start_string)] == start_string):
                                line = line[len(start_string):]
                                caption = ''
                                while(line[-len(caption_end_string):]
                                      != caption_end_string):
                                    caption += line.strip()
                                    line = f.readline()
                                    line_number += 1
                                caption += line[:-len(caption_end_string)]
                                caption = caption.rstrip(
                                    string.whitespace + '.'
                                )
                                self._captions[img_path].append(caption)

        # End of caption file preprocessing.
        self._vocabulary, _ = self.get_vocabulary()


class Flickr(ImageCaptionDataset):
    def __init__(
        self,
        caption_filename='',
        data_dir='./',
    ):
        super().__init__(data_dir)
        prev_img_id = None

        with open(caption_filename) as f:
            for line in f:
                filename_length = line.find('#')
                img_id = line[:filename_length]
                if img_id != prev_img_id:
                    self._captions[img_id] = []
                    self._img_ids.append(img_id)
                    prev_img_id = img_id

                raw_caption = line[filename_length + 3:]
                caption = raw_caption.rstrip(
                    string.whitespace + '.'
                )
                self._captions[img_id].append(caption)
            
