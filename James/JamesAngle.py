import numpy as np
from JamesPredict import predict
from colorama import Fore


class JamesAngle:
	def __init__(self, image: str, model: str):
		self.prediction = predict(image, model, True)
		self.person = self.__get_person()
		if self.person:
			self.left_arm, self.right_arm, self.hips, self.core, self.left_leg, self.right_leg, self.is_facing = self.__get_joints()
			self.left_range = range(50, 110, 1)
			self.right_range = range(50, 110, 1)
			self.forward_range = range(110, 160, 1)
			self.reverse_range = range(110, 160, 1)

	def __get_person(self):
		"""
			Returns there is a person in the image.
		"""
		counter = 0
		for i in range(15):
			if self.prediction[0][i][0] < 20 and self.prediction[0][i][1] < 20:
				counter += 1
		if counter > 5:
			return False
		else:
			return True

	def __get_joints(self):
		"""
			Returns the all joints except head, and checks which way the person is facing.
		"""
		left_arm, right_arm, hips, core, left_leg, right_leg = [], [], [], [], [], []

		# print(self.prediction[0])
		left_arm.append(self.prediction[0][5])
		left_arm.append(self.prediction[0][2])
		left_arm.append(self.prediction[0][6])
		right_arm.append(self.prediction[0][12])
		right_arm.append(self.prediction[0][9])
		right_arm.append(self.prediction[0][13])

		hips.append(self.prediction[0][4])
		hips.append(self.prediction[0][11])

		core.append(self.prediction[0][7])
		core.append(self.prediction[0][14])

		left_leg.append(self.prediction[0][4])
		left_leg.append(self.prediction[0][1])

		right_leg.append(self.prediction[0][11])
		right_leg.append(self.prediction[0][8])

		if self.prediction[0][5][0] > self.prediction[0][12][0]:
			return left_arm, right_arm, hips,  core, left_leg, right_leg, True
		else:
			return left_arm, right_arm, hips,  core, left_leg, right_leg, False

	def __get_signal(self):
		"""
			Returns the signal that the person is facing.
			Returns: str
		"""
		if self.person:
			left_arm_angle = self.__calculate_angle(self.left_arm[0], self.left_arm[1], self.left_arm[2])
			right_arm_angle = self.__calculate_angle(self.right_arm[0], self.right_arm[1], self.right_arm[2])

			if (self.hips[0][0] < 10 and self.hips[0][1] < 10) or (self.hips[1][0] < 10 and self.hips[1][1] < 10):
				left_signal_angle = 0
				right_signal_angle = 0
			elif self.left_arm[0][0] < 10 and self.left_arm[0][1] < 10 and self.left_arm[1][0] < 10 and self.left_arm[1][1] < 10:
				left_signal_angle = 0
				right_signal_angle = 0
			elif self.right_arm[0][0] < 10 and self.right_arm[0][1] < 10 and self.right_arm[1][0] < 10 and self.right_arm[1][1] < 10:
				left_signal_angle = 0
				right_signal_angle = 0
			else:
				left_signal_angle = self.__calculate_angle(self.hips[0], self.left_arm[0], self.left_arm[1])
				right_signal_angle = self.__calculate_angle(self.hips[1], self.right_arm[0], self.right_arm[1])

			if self.is_facing:
				if int(float(left_signal_angle)) in self.left_range and int(float(right_signal_angle)) in self.right_range:
					signal = 'stop'
				elif int(float(left_signal_angle)) in self.reverse_range and int(float(right_signal_angle)) in self.reverse_range:
					signal = 'reverse'
				elif int(float(left_signal_angle)) in self.forward_range or int(float(right_signal_angle)) in self.forward_range:
					signal = 'forward'
				elif int(float(left_signal_angle)) in self.left_range and int(float(right_signal_angle)) not in self.right_range:
					signal = 'right'
				elif int(float(left_signal_angle)) not in self.left_range and int(float(right_signal_angle)) in self.right_range:
					signal = 'left'
				else:
					print(Fore.CYAN, 'HIT ELSE')
					signal = 'stop'
			else:
				if int(float(left_signal_angle)) in self.left_range and int(float(right_signal_angle)) in self.right_range:
					signal = 'stop'
				elif int(float(left_signal_angle)) in self.reverse_range and int(float(right_signal_angle)) in self.reverse_range:
					signal = 'reverse'
				elif int(float(left_signal_angle)) in self.forward_range or int(float(right_signal_angle)) in self.forward_range:
					signal = 'forward'
				elif int(float(left_signal_angle)) in self.left_range and int(float(right_signal_angle)) not in self.right_range:
					signal = 'left'
				elif int(float(left_signal_angle)) not in self.left_range and int(float(right_signal_angle)) in self.right_range:
					signal = 'right'
				else:
					print(Fore.CYAN, 'HIT ELSE')
					signal = 'stop'

			self.print_signal_info(left_arm_angle, right_arm_angle, left_signal_angle, right_signal_angle, signal)
		else:

			print(Fore.RED, 'Signal: stop, no person in the image!')

	def print_signal_info(self, left_arm_angle, right_arm_angle, left_signal_angle, right_signal_angle, signal):
		print(Fore.YELLOW, '---------------------Signal data ---------------------')
		print(Fore.RED, self.right_arm)
		print(Fore.GREEN, self.left_arm)
		print(Fore.BLUE, f'Person is facing camera: {self.is_facing}\n')

		# print(Fore.RED, f'Right arm angle: {right_arm_angle}')
		# print(Fore.GREEN, f'Left arm angle: {left_arm_angle}\n')

		print(Fore.RED, f'Right signal angle: {right_signal_angle}')
		print(Fore.GREEN, f'Left signal angle: {left_signal_angle}\n')

		print(Fore.MAGENTA, f'Signal: {signal}')
		print(Fore.YELLOW, '-----------------------------------------------------')

	def __calculate_angle(self, point1, point2, point3):
		"""
			Calculates the angle between three points.
			"""
		a = np.array(point1)
		b = np.array(point2)
		c = np.array(point3)
		ba = a - b  # Difference between a and b
		bc = c - b  # Difference between c and b

		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # cosine of the angle
		angle = np.arccos(cosine_angle)  # angle in radians
		degrees = np.degrees(angle)  # angle in degrees
		if degrees is None:
			return 0
		else:
			return degrees

	def run(self):
		self.__get_signal()
