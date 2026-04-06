from utils import ndarray_to_base64, model_selection


class Agents:
  def __init__(self, image, task_description, model: str = "gpt-4o"):
    self.image = image
    self.encoded_image = ndarray_to_base64(image)
    self.task_description = task_description
    self.client, self.model = model_selection(model)

  def multi_agent_vision_planning(self, objs_from_scene=None):
    def filter_task_objects(objs):
      if not objs:
        return objs

      task_words = set(self.task_description.lower().replace("-", " ").split())
      always_keep = {"sphere", "ball", "table", "workspace", "robot", "panda", "arm"}

      filtered = []
      for obj in objs:
        if not isinstance(obj, dict):
          continue
        name = str(obj.get("name", ""))
        name_tokens = set(name.lower().replace("_", " ").replace("-", " ").split())
        if name_tokens & task_words:
          filtered.append(obj)
          continue
        if name_tokens & always_keep:
          filtered.append(obj)

      # If filtering is too strict, fallback to original list.
      return filtered if len(filtered) > 0 else objs

    task_objs = filter_task_objects(objs_from_scene)

    def environment_agent():
      """提取其中的主要物体及其空间关系，并以(主体, 关系, 客体)的三元组格式返回描述。"""
      agent = self.client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "You are an assistent which is able accurately describe the content of an image. \n\
                        In particular, you are able to capture the main objects present.\n\
                        Explore the image accurately as an expert and find all the object that you can see.\n\
                        in the image and provide the relations that exist between them. \n\
                        These relations are described in the form of a triple (subject, relation, object) \
                        and when you answer you are only expected to answer with triples and nothing else. \n\
                        When writing the triples, try to execute this task: " + self.task_description + "\n\
                        and verify the elements that you neeed to solve and write the relation of the objects in the image.\n\
                        For example, if in a scene there is a door, a table in front of the door and a book on the table \
                        with a pen right to it, your answer should be: \
                        1) (table, in front of, door) \n\
                        2) (book, on, table) \n\
                        3) (pen, on, table) \n\
                        4) (pen, right to, book) \n\
                        5) (book, left to, pen). \n\
                        Only mention objects that are necessary for the current task.\n\
                        Do not introduce irrelevant objects even if they exist in scene metadata.\n\
                        At the end of the task, you must write a instruction to solve the task, in a way that you can\
                        help who read your answer to understand how to solve the task without knowing the scene.",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
              },
            ],
          }
        ],
        max_tokens=500,
        temperature=0,
      )

      response = agent.choices[0].message.content
      print("[environment_agent] response:", response)
      return response

    def sim_ground_agent(objs_from_scene=None):
      """在给定场景物体先验知识（objs_from_scene）的约束下，生成一份详尽、无歧义的自然语言场景描述。"""
      prompt = (
        f"Here is the oracle objects involved in the task: \n{task_objs}\n"
        "Do not use any objects not in the scene. "
        "Only mention objects strictly relevant to the task goal."
      )
      agent = self.client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "You are an assistent which is able accurately describe the content of an image. \n\
                            In particular, you are able to describe accurately the content of the image to make one understand \
                            all the details of the image without seeing it. \n\
                            You should describe how the scene it is made with high level description and precise instruction to solve\
                            the following task : " + self.task_description + prompt + "\n\
                            If the task contains ambiguity in the solution of the task , for example same objects of the same type,\
                            specify the position of the object in the image or in relation at other objects.\n",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
              },
            ],
          }
        ],
        max_tokens=500,
        temperature=0,
      )

      response = agent.choices[0].message.content
      print("[sim_ground_agent] response:", response)
      return response

    enviroment_info = environment_agent() + "\n" + sim_ground_agent(task_objs)
    agent = self.client.chat.completions.create(
      model=self.model,
      messages=[
        {
          "role": "system",
          "content": "You are an  helpful assistant which is able accurately describe the navigation planning step to reach the required goal.\n\
             You know how are the object that you can use and where are from the following information "
          + enviroment_info
          + "\
             You will do a planning to execute the goal using the information written.\n\
            Use only task-relevant objects and avoid unrelated distractors.\n\
            Your answer will be a list of only steps that help a agent to reach the goal. Try to do precise information for each step but in atomic way\n\
            Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                and containing only the atomic step with the position of the object and nothing else.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using the objects \
                    perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n",
        },
        {"role": "user", "content": "The goal is " + self.task_description},
      ],
      temperature=0,
      max_tokens=300,
    )
    print("[planning_agent] response:", agent.choices[0].message.content)
    return enviroment_info, agent.choices[0].message.content
