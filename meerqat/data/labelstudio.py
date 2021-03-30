# coding: utf-8

def retrieve_vqa(completion):
    results = completion["completions"][0]["result"]
    data = completion["data"]
    vqa = dict(question=data["question"], wikidata_id=data["wikidata_id"], answer=data['answer'], image=data['image'])
    # make a proper dict out of the results
    # note that "vq" is always present in results, even if the user didn't modify it
    for result in results:
        key = result["from_name"]
        vqa[key] = next(iter(result["value"].values()))[0]

    # update image if necessary
    change_image = vqa.pop("change_image", None)
    if change_image is not None:
        # e.g. "$altimage1caption" -> "altimage1"
        vqa['image'] = data[change_image[1: -7]]

    return vqa