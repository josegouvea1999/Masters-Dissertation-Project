import { Text, ScrollView, View, Image, Pressable, Modal, ActivityIndicator, Animated, Easing, Linking } from "react-native";
import React ,{useEffect, useState, useRef} from "react";
import {styles} from "./Styles"
import { LinearGradient } from "expo-linear-gradient";
import axios from "axios";

const SERVER_URL = "http://192.168.1.70:5000"

const PulsatingImage = () => {

  const scaleValue = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const pulse = () => {
      Animated.sequence([
        Animated.timing(scaleValue, {
          toValue: 1.2,
          duration: 1000,
          easing: Easing.linear,
          useNativeDriver: true,
        }),
        Animated.timing(scaleValue, {
          toValue: 1,
          duration: 1000,
          easing: Easing.linear,
          useNativeDriver: true,
        }),
      ]).start(() => pulse());
    };

    pulse();
  }, [scaleValue]);

  return (
    <View style={styles.overlayContainer}>
      <Animated.Image
        source={require('../assets/logo_filled.png')}
        style={[styles.pulsatingImage, { transform: [{ scale: scaleValue }] }]}
      />
    </View>
  );
};

const AddScreen = ({ route }) => {

  const { playlists, playlistCovers } = route.params;

  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [radio_url, setRadioUrl] = useState(null);

  const [modalVisible, setModalVisible] = useState(false);
  const [genInProgress, setGenInProgress] = useState(false);
  const [genCompleted, setGenCompleted] = useState(false);
  const [modalHeight, setModalHeight] = useState(300);

  const setModal = async (playlist_id, playlist_name, playlist_cover, playlist_url) => {

    setSelectedPlaylist({playlist_id, playlist_name, playlist_cover, playlist_url});
    setModalVisible(true);
  }

  const genRecommendations = async () => {
    try {
      setModalHeight(600);
      setGenInProgress(true);

      console.log("Generating radio for playlist: " + selectedPlaylist.playlist_name);

      const response = await axios.get( SERVER_URL + "/refresh/" + selectedPlaylist.playlist_id, {withCredentials : true} );
      
      const data = await response.data;

      setRadioUrl(data.external_urls.spotify);
      setGenInProgress(false);
      setGenCompleted(true);

      console.log("Radio as been successfully generated for playlist " + selectedPlaylist.playlist_name);
      
    } catch (error) {
      console.log(error);
    }
  }

  return (  
    <LinearGradient colors={["#040306", "#131624"]} style={{ flex: 1 }}>
 
        <View>
          <View style={styles.userBar}>
              <Text style={[styles.userMessage, {color: "white"}]}>
                {"Please select the playlist you wish to expand..."}
              </Text>
        </View>

        <ScrollView style={{ marginTop: 10, marginBottom: 150 }}>

          <Modal
            animationType="slide"
            transparent={true}
            visible={modalVisible}
            onRequestClose={() => setModalVisible(!modalVisible)}>
            <View style={styles.centeredView}>
              <LinearGradient colors={["#1DB954", "#131624"]} style={[styles.modalView, {height: modalHeight}]}>
                {genInProgress ? (
                      selectedPlaylist && (
                          <View style={styles.normalContent}>
                              <View style={[{flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: 100}]}>
                                  <Image source={{ uri: selectedPlaylist.playlist_cover }} style={styles.playlistImageModalExpanding} />
                                  <PulsatingImage style={{position: 'absolute', left:'50%'}}/>
                              </View>
                              <Text style={[styles.textStyleExpanding, styles.modalTextExpanding]}>
                                  {`Generating RADIO for "${selectedPlaylist.playlist_name}" playlist...`}
                              </Text>
                              <ActivityIndicator size="large" color="white"/>
                          </View>
                      )
                  ) : (
                      genCompleted ? selectedPlaylist && (
                          <View style={styles.normalContent}>
                              <Text style={styles.textStyleSuccess}>
                                  {"RECOMMENDATIONS GENERATED SUCCESSFULLY!"}
                              </Text>
                              <Text style={styles.textStyleSuccess2}>
                                  Listen to your new recommendations in the designated [RADIO] playlist in your Spotify library:
                              </Text>
                              <Image source={require('../assets/logo.jpg')} style={styles.playlistImageModalSuccess} />
                              <Text style={styles.textStyleSuccess3}>
                                  {selectedPlaylist.playlist_name} [RADIO]
                              </Text>
                              <View style={styles.buttonContainerSuccess}>
                                  <Pressable
                                      style={[
                                          styles.buttonSuccess,
                                          styles.buttonClose,
                                          styles.confirmButtonSuccess,
                                          ]}
                                      onPress={() => Linking.openURL(radio_url)}>
                                      <Text style={[styles.textStyle, styles.confirmButtonText]}>Go to playlist [RADIO]</Text>
                                  </Pressable>
                                  <Pressable
                                      style={[
                                          styles.buttonSuccess,
                                          styles.buttonClose,
                                          styles.cancelButton,
                                      ]}
                                      onPress={() => setModalVisible(false)}>
                                      <Text style={[styles.textStyle, {fontSize: 14}]}>Get more recommendations...</Text>
                                  </Pressable>
                              </View>
                          </View>
                      ) : (
                          selectedPlaylist && (
                              <View style={styles.normalContent}>
                                  <Text style={[styles.textStyle, styles.modalText]}>
                                      {`Do you wish to generate a RADIO for this playlist?`}
                                  </Text>
                                  <View style={styles.playlistInfoContainer}>
                                      <Image
                                          source={{ uri: selectedPlaylist.playlist_cover }}
                                          style={styles.playlistImageModal}/>
                                      <Text style={styles.playlistNameModal}>
                                          {selectedPlaylist.playlist_name}
                                      </Text>
                                  </View>
                                  <View style={styles.buttonContainer}>
                                      <Pressable
                                          style={[
                                              styles.button,
                                              styles.buttonClose,
                                              styles.cancelButton,
                                          ]}
                                          onPress={() => setModalVisible(!modalVisible)}>
                                          <Text style={[styles.textStyle, styles.cancelButtonText]}>Cancel</Text>
                                      </Pressable>
                                      <Pressable
                                          style={[
                                              styles.button,
                                              styles.buttonClose,
                                              styles.confirmButton,
                                          ]}
                                          onPress={() => genRecommendations()}>
                                          <Text style={[styles.textStyle, styles.confirmButtonText]}>Confirm</Text>
                                      </Pressable>
                                  </View>
                              </View>
                          )
                      )
                  )}
              </LinearGradient>
            </View>
          </Modal>

          {playlists.map((playlist, i) => (
            <Pressable
              key={i}
              onPress={() => setModal(playlist.id, playlist.name, playlistCovers[i], playlist.external_urls.spotify)}
              style={styles.playlistPressable}>
              <LinearGradient colors={["#33006F", "#FFFFFF"]}>
                <Pressable style={styles.coverPressable}>
                  <Image
                    source={{ uri: playlistCovers[i]  }}
                    style={{ width: 75, height: 75 }}/>
                </Pressable>
              </LinearGradient>

              <Text style={{ color: "white", fontSize: 15, fontWeight: "bold", maxWidth: '80%' }}>
                {playlist.name}
              </Text>
            </Pressable>))}

        </ScrollView>
      </View>

    </LinearGradient>
  );
};

export default AddScreen;